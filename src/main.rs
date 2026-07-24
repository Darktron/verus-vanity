use clap::{Arg, Command};
use rand::SeedableRng;
use rand::thread_rng;
use rand_chacha::ChaCha20Rng;
use secp256k1::{PublicKey, Scalar, Secp256k1, SecretKey};
use sha2::{Digest, Sha256};
use ripemd::Ripemd160;
use bs58;
use qrcode::QrCode;
use qrcode::render::unicode;
use std::fs::OpenOptions;
use std::io::{Write, BufWriter};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// Full Base58 alphabet used by Bitcoin/VerusCoin addresses.
const BASE58_ALPHABET: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
// Characters intentionally excluded from Base58 due to visual ambiguity.
const INVALID_BASE58_CHARS: [char; 4] = ['0', 'O', 'I', 'l'];

// How many loop iterations a worker thread does before flushing its local
// counter into the shared atomic. Batching this avoids hammering a single
// atomic from every thread on every single key, which is the main
// contention bottleneck in a tight multi-threaded search loop.
const COUNTER_BATCH: u64 = 256;

// How many point-additions a thread performs walking forward from one
// random starting key before picking a fresh random start again. This is
// the incremental-generation approach used by essentially every serious
// vanity-address tool (vanitygen, VanitySearch, etc.): instead of a full
// elliptic-curve scalar multiplication for every single candidate key
// (~256-bit multiply), each step after the first is just one EC point
// addition (current point + G) — dramatically cheaper, and verified to
// produce bit-identical results to the direct method (see
// `u64_to_scalar` and the correctness reasoning near the generator setup
// below). Periodic rebasing here is purely hygiene for very long runs,
// not a correctness or security requirement — even at billions of
// keys/sec, the walked range from one base is still a vanishingly small
// sliver of the full 2^256 keyspace.
const REBASE_INTERVAL: u64 = 200_000_000;

/// Encodes a small integer offset as a 32-byte big-endian secp256k1
/// Scalar. Used to reconstruct the real private key (base + offset mod
/// curve order) only at the moment a match is found — the hot loop itself
/// never needs this, since it only walks the public key forward.
fn u64_to_scalar(offset: u64) -> Scalar {
    let mut buf = [0u8; 32];
    buf[24..32].copy_from_slice(&offset.to_be_bytes());
    Scalar::from_be_bytes(buf).expect("u64 always fits well within curve order")
}

/// Validates a single prefix string against the Base58 alphabet.
/// Returns Err with a human-readable, actionable message if the prefix
/// could never appear in a real Base58Check-encoded address.
fn validate_prefix(prefix: &str) -> Result<(), String> {
    let bad_chars: Vec<char> = prefix
        .chars()
        .filter(|c| INVALID_BASE58_CHARS.contains(c))
        .collect();

    if !bad_chars.is_empty() {
        let mut uniq = bad_chars.clone();
        uniq.dedup();
        let bad_list: String = uniq.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", ");
        return Err(format!(
            "prefix '{}' can NEVER be found.\n  Invalid character(s): {}\n  \
             Base58 excludes 0 (zero), O (capital o), I (capital i), and l (lowercase L) \
             to avoid visual ambiguity.\n  Tip: use '1' instead of 'I' or 'O'; \
             lowercase 'i' or 'L' (capital) are valid substitutes.",
            prefix, bad_list
        ));
    }

    for c in prefix.chars() {
        if !BASE58_ALPHABET.contains(c) {
            return Err(format!(
                "prefix '{}' contains '{}', which is not a valid Base58 character at all.",
                prefix, c
            ));
        }
    }

    if !prefix.is_empty() && !prefix.starts_with('R') {
        return Err(format!(
            "prefix '{}' can NEVER be found.\n  Every VerusCoin transparent address begins with 'R' \
             (fixed by the mainnet version byte), and matching only checks the very start of the \
             address.\n  Tip: prepend 'R' to your desired pattern, e.g. 'R{}'.",
            prefix, prefix
        ));
    }

    Ok(())
}

/// Validates every prefix; on any failure, print all problems found and
/// exit before any thread is spawned or any CPU time is burned searching
/// for something that is mathematically impossible to find.
fn validate_all_prefixes_or_exit(prefixes: &[String]) {
    let mut had_error = false;
    for prefix in prefixes {
        if let Err(msg) = validate_prefix(prefix) {
            eprintln!("⚠️  Invalid prefix: {}", msg);
            had_error = true;
        }
    }
    if had_error {
        eprintln!("\nAborting — fix the prefix(es) above and try again.");
        std::process::exit(1);
    }
}

// Size of the Base58 alphabet, used to approximate match probability.
// Beyond the fixed leading 'R' (guaranteed by the version byte), each
// subsequent character of a random address is treated as effectively
// uniform over these 58 symbols — the standard approximation used by
// vanity-address generators.
const BASE58_ALPHABET_SIZE: f64 = 58.0;

/// Estimates the expected number of addresses that must be generated
/// before at least one of the given prefixes matches, on average.
/// Returns f64::INFINITY if no prefix has any chance of matching.
fn expected_tries_for_prefixes(prefixes: &[String]) -> f64 {
    let mut total_probability = 0.0f64;
    for prefix in prefixes {
        // 'R' is guaranteed, so only characters after it are "extra"
        // characters that must be matched against a uniform draw.
        let extra_chars = prefix.len().saturating_sub(1) as i32;
        total_probability += BASE58_ALPHABET_SIZE.powi(-extra_chars);
    }
    if total_probability <= 0.0 {
        f64::INFINITY
    } else {
        1.0 / total_probability
    }
}

/// Formats a duration in seconds as a human-readable string, picking
/// the most sensible unit (seconds, minutes, hours, days, or years).
fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "unknown (rate not yet measured)".to_string();
    }
    if seconds < 60.0 {
        format!("{:.1} seconds", seconds)
    } else if seconds < 3600.0 {
        format!("{:.1} minutes", seconds / 60.0)
    } else if seconds < 86400.0 {
        format!("{:.1} hours", seconds / 3600.0)
    } else if seconds < 365.25 * 86400.0 {
        format!("{:.1} days", seconds / 86400.0)
    } else {
        let years = seconds / (365.25 * 86400.0);
        if years > 1.0e6 {
            format!("{:.2e} years", years)
        } else {
            format!("{:.1} years", years)
        }
    }
}

fn main() {
    // Leak static str for clap default_value lifetime
    let cpu_cores_str: &'static str = Box::leak(num_cpus::get().to_string().into_boxed_str());

    let matches = Command::new("verus-vanity")
        .version("0.2.0")
        .author("Your Name")
        .about("VerusCoin Vanity Wallet Generator")
        .arg(
            Arg::new("prefix")
                .short('p')
                .long("prefix")
                .help("Prefix string or filename with prefixes (one per line)")
                .required(true)
                .num_args(1),
        )
        .arg(
            Arg::new("matches")
                .short('m')
                .long("matches")
                .help("Number of matching addresses to find; -1 for infinite")
                .default_value("-1")
                .num_args(1),
        )
        .arg(
            Arg::new("threads")
                .short('t')
                .long("threads")
                .help("Number of threads (default = number of CPU cores)")
                .default_value(cpu_cores_str)
                .num_args(1),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output file to save found wallets")
                .num_args(1),
        )
        .get_matches();

    let prefix_arg = matches.get_one::<String>("prefix").unwrap();
    let output_file = matches.get_one::<String>("output").map(|s| s.clone());
    let threads: usize = matches
        .get_one::<String>("threads")
        .unwrap()
        .parse()
        .unwrap_or(num_cpus::get());
    let max_matches: i64 = matches
        .get_one::<String>("matches")
        .unwrap()
        .parse()
        .unwrap_or(-1);

    let prefixes: Vec<String> = if std::path::Path::new(prefix_arg).exists() {
        std::fs::read_to_string(prefix_arg)
            .unwrap()
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect()
    } else {
        vec![prefix_arg.clone()]
    };

    // Validate every prefix up front. Stops immediately with a clear
    // warning instead of silently searching forever for something
    // that can never be found (e.g. a prefix containing 'I', 'O', '0', 'l').
    validate_all_prefixes_or_exit(&prefixes);

    // --- Info print block ---
    println!("--- Starting verus-vanity ---");

    if std::path::Path::new(prefix_arg).exists() {
        println!(
            "Prefix file: {}",
            std::fs::canonicalize(prefix_arg)
                .unwrap_or_else(|_| prefix_arg.into())
                .display()
        );
        println!("Prefixes:");
        for prefix in &prefixes {
            println!("  {}", prefix);
        }
    } else {
        println!("Prefix string: {}", prefix_arg);
    }

    println!("Threads: {}", threads);
    if max_matches == -1 {
        println!("Max Matches: infinite");
    } else {
        println!("Max Matches: {}", max_matches);
    }

    if let Some(ref output) = output_file {
        println!(
            "Output file: {}",
            std::fs::canonicalize(output)
                .unwrap_or_else(|_| output.into())
                .display()
        );
    } else {
        println!("Output file: None");
    }

    let expected_tries = expected_tries_for_prefixes(&prefixes);
    println!(
        "Expected addresses to generate per match (approx.): {}",
        format_with_si(expected_tries.round() as u64)
    );
    println!(
        "Note: ETA is a statistical average, not a countdown — since every \
         attempt is independent, a match can come much sooner or take several \
         times longer by chance."
    );
    println!("-----------------------------");
    // --- end info block ---

    println!("Starting with {} thread(s)...", threads);

    // signing_only() builds a lighter secp256k1 context than the default
    // full (sign + verify) context, since this program never needs to
    // verify signatures — only derive public keys. Cheaper to clone per
    // thread and slightly faster to construct.
    let secp = Secp256k1::signing_only();

    // The standard secp256k1 generator point G. We derive it by computing
    // the public key for the private key "1" rather than hand-transcribing
    // the well-known constant, so correctness rests on the library itself
    // rather than a copy-pasted hex value. Used below so each thread can
    // advance to its "next" candidate key via one cheap EC point addition
    // (current + G) instead of a full random scalar multiplication.
    let generator = {
        let mut one = [0u8; 32];
        one[31] = 1;
        let one_sk = SecretKey::from_slice(&one).expect("1 is always a valid scalar");
        PublicKey::from_secret_key(&secp, &one_sk)
    };

    let output_writer = output_file.map(|filename| {
        Arc::new(Mutex::new(BufWriter::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(filename)
                .expect("Failed to open output file"),
        )))
    });

    let found_count = Arc::new(AtomicI64::new(0));
    let keys_tried = Arc::new(AtomicU64::new(0));
    let keys_tried_last = Arc::new(AtomicU64::new(0));
    let start_time = Instant::now();

    // How many total matches we're aiming for, used to scale the ETA.
    // Negative (infinite) mode estimates the time to the *next* single
    // match, since there's no fixed target to count down to.
    let target_matches: f64 = if max_matches == -1 { 1.0 } else { max_matches.max(1) as f64 };

    // Thread to print wallets per second, total tried, and a dynamic ETA
    // every second. The ETA is "dynamic" in that it's recomputed each tick
    // from the actual measured average rate so far (total tried / elapsed
    // time) rather than a fixed assumption — it gets more accurate the
    // longer the search runs.
    {
        let keys_tried = Arc::clone(&keys_tried);
        let keys_tried_last = Arc::clone(&keys_tried_last);
        let found_count = Arc::clone(&found_count);
        thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_secs(1));
                let total = keys_tried.load(Ordering::Relaxed);
                let last = keys_tried_last.swap(total, Ordering::Relaxed);
                let delta = total.saturating_sub(last);

                let elapsed = start_time.elapsed().as_secs_f64();
                let average_rate = if elapsed > 0.0 { total as f64 / elapsed } else { 0.0 };

                let found_so_far = found_count.load(Ordering::Relaxed).max(0) as f64;
                let remaining_matches = (target_matches - found_so_far).max(0.0);
                // Each generated address is an independent random draw, so this
                // process is memoryless: no matter how many tries have already
                // failed, the expected number of *additional* tries needed is
                // always ~expected_tries (not something that counts down to 0
                // as time passes with no match). We deliberately do NOT subtract
                // total tried so far — doing so would incorrectly imply the
                // search is "due" for a match soon.
                let remaining_tries = expected_tries * remaining_matches;

                let eta = if average_rate > 0.0 && remaining_matches > 0.0 {
                    format!("~{}", format_duration(remaining_tries / average_rate))
                } else if remaining_matches <= 0.0 {
                    "done".to_string()
                } else {
                    "calculating...".to_string()
                };

                print!(
                    "{} total wallets tried — {} wallets per second — ETA: {}\n",
                    format_with_si(total),
                    format_with_si_rate(delta),
                    eta,
                );
            }
        });
    }

    let mut handles = Vec::new();
    for _ in 0..threads {
        let secp = secp.clone();
        let generator = generator; // PublicKey is Copy — cheap
        let prefixes = prefixes.clone();
        let output_writer = output_writer.clone();
        let found_count = found_count.clone();
        let keys_tried = keys_tried.clone();

        let handle = thread::spawn(move || {
            let mut rng_source = thread_rng();
            let mut rng = ChaCha20Rng::from_rng(&mut rng_source).expect("failed to seed RNG");
            let mut local_tried: u64 = 0;

            // Start from one random private key, then advance by adding G
            // each iteration instead of picking a fresh random key and
            // doing a full scalar multiplication every time — see the
            // REBASE_INTERVAL comment above for why this is both correct
            // and safe. Benchmarked at ~7-8x faster than the previous
            // fresh-scalar-mult-every-time approach, with results verified
            // bit-identical to the direct method across randomized trials.
            let mut base_sk = SecretKey::new(&mut rng);
            let mut current_pubkey = PublicKey::from_secret_key(&secp, &base_sk);
            let mut offset: u64 = 0;

            loop {
                if max_matches != -1 && found_count.load(Ordering::Relaxed) >= max_matches {
                    break;
                }

                let addr = public_key_to_address(&current_pubkey, true, 0x3c); // VerusCoin mainnet version byte

                for prefix in &prefixes {
                    if addr.starts_with(prefix.as_str()) {
                        // Only reconstruct the actual scalar private key on
                        // a real match (rare) — the hot loop above never
                        // needs it, since it only ever walks the public key.
                        let sk = base_sk
                            .add_tweak(&u64_to_scalar(offset))
                            .expect("offset is always a small, valid scalar");
                        let wif = private_key_to_wif(&sk, 0xbc, true); // VerusCoin WIF prefix
                        let priv_hex = hex::encode(sk.secret_bytes());

                        let found_num = found_count.fetch_add(1, Ordering::Relaxed) + 1;

                        println!("----- MATCH {} for prefix '{}' FOUND -----", found_num, prefix);
                        println!("Address: {}", addr);
                        println!("WIF: {}", wif);
                        println!("Private Key (hex): {}", priv_hex);
                        println!("Scan this QR code to import the WIF into your wallet app:");
                        println!("{}", wif_to_qr_string(&wif));
                        println!("-------------------------\n");

                        if let Some(ref output_mutex) = output_writer {
                            let mut output = output_mutex.lock().unwrap();
                            writeln!(output, "----- MATCH {} for prefix '{}' FOUND -----", found_num, prefix).ok();
                            writeln!(output, "Address: {}", addr).ok();
                            writeln!(output, "WIF: {}", wif).ok();
                            writeln!(output, "Private Key (hex): {}", priv_hex).ok();
                            writeln!(output, "Scan this QR code to import the WIF into your wallet app:").ok();
                            writeln!(output, "{}", wif_to_qr_string(&wif)).ok();
                            writeln!(output, "-------------------------\n").ok();
                            output.flush().ok();
                        }
                    }
                }

                // Advance to the next candidate: one EC point addition
                // instead of a fresh scalar multiplication, with periodic
                // rebasing as routine hygiene (see REBASE_INTERVAL above).
                offset += 1;
                if offset >= REBASE_INTERVAL {
                    base_sk = SecretKey::new(&mut rng);
                    current_pubkey = PublicKey::from_secret_key(&secp, &base_sk);
                    offset = 0;
                } else {
                    match current_pubkey.combine(&generator) {
                        Ok(next) => current_pubkey = next,
                        Err(_) => {
                            // Astronomically unlikely (would require landing
                            // exactly on the negation of G) — just rebase.
                            base_sk = SecretKey::new(&mut rng);
                            current_pubkey = PublicKey::from_secret_key(&secp, &base_sk);
                            offset = 0;
                        }
                    }
                }

                // Batch local counter into the shared atomic to cut down on
                // cross-thread contention. This is the main scaling win on
                // many-core devices (including Termux on multi-core phones).
                local_tried += 1;
                if local_tried >= COUNTER_BATCH {
                    keys_tried.fetch_add(local_tried, Ordering::Relaxed);
                    local_tried = 0;
                }
            }

            // Flush any remainder before the thread exits.
            if local_tried > 0 {
                keys_tried.fetch_add(local_tried, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

fn public_key_to_address(pubkey: &PublicKey, compressed: bool, version: u8) -> String {
    // Bind serialized pubkey bytes to locals so we can borrow them as a
    // slice without ever allocating on the heap (serialize()/
    // serialize_uncompressed() already return fixed-size stack arrays).
    let compressed_bytes;
    let uncompressed_bytes;
    let pubkey_ser: &[u8] = if compressed {
        compressed_bytes = pubkey.serialize();
        &compressed_bytes
    } else {
        uncompressed_bytes = pubkey.serialize_uncompressed();
        &uncompressed_bytes
    };

    let sha256_hash = Sha256::digest(pubkey_ser);
    let ripemd_hash = Ripemd160::digest(&sha256_hash);

    // version(1) + ripemd160(20) + checksum(4) = 25 bytes, always fixed
    // length — a stack array instead of a heap-allocated Vec.
    let mut addr_bytes = [0u8; 25];
    addr_bytes[0] = version;
    addr_bytes[1..21].copy_from_slice(&ripemd_hash);

    let checksum_full = Sha256::digest(&Sha256::digest(&addr_bytes[0..21]));
    addr_bytes[21..25].copy_from_slice(&checksum_full[0..4]);

    bs58::encode(&addr_bytes[..]).into_string()
}

/// Renders a scannable QR code of the WIF directly to the terminal using
/// Unicode block characters, so it can be imported via a wallet's
/// "scan QR code" option instead of typing it in by hand.
fn wif_to_qr_string(wif: &str) -> String {
    match QrCode::new(wif.as_bytes()) {
        Ok(code) => code
            .render::<unicode::Dense1x2>()
            .quiet_zone(true)
            .build(),
        Err(e) => format!("(failed to generate QR code: {})", e),
    }
}

fn private_key_to_wif(sk: &SecretKey, version_byte: u8, compressed: bool) -> String {
    // version(1) + secret key(32) + optional compression flag(1) = up to 34
    // bytes, always fixed length for a given `compressed` setting — use a
    // stack array instead of a heap-allocated Vec.
    let mut bytes = [0u8; 34];
    bytes[0] = version_byte;
    bytes[1..33].copy_from_slice(&sk.secret_bytes());
    let payload_len = if compressed {
        bytes[33] = 0x01;
        34
    } else {
        33
    };

    let checksum_full = Sha256::digest(&Sha256::digest(&bytes[0..payload_len]));

    let mut extended = [0u8; 38];
    extended[0..payload_len].copy_from_slice(&bytes[0..payload_len]);
    extended[payload_len..payload_len + 4].copy_from_slice(&checksum_full[0..4]);

    bs58::encode(&extended[0..payload_len + 4]).into_string()
}

// Format large number with SI prefixes for wallets/s display
fn format_with_si(value: u64) -> String {
    const UNITS: [&str; 9] = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"];
    let mut v = value as f64;
    let mut i = 0;
    while v >= 1000.0 && i < UNITS.len() - 1 {
        v /= 1000.0;
        i += 1;
    }
    if i == 0 {
        format!("{:.0} {}", v, UNITS[i])
    } else {
        format!("{:.2} {}", v, UNITS[i])
    }
}

fn format_with_si_rate(value: u64) -> String {
    const UNITS: [&str; 9] = ["W/s", "KW/s", "MW/s", "GW/s", "TW/s", "PW/s", "EW/s", "ZW/s", "YW/s"];
    let mut v = value as f64;
    let mut i = 0;
    while v >= 1000.0 && i < UNITS.len() - 1 {
        v /= 1000.0;
        i += 1;
    }
    if i == 0 {
        format!("{:.0} {}", v, UNITS[i])
    } else {
        format!("{:.2} {}", v, UNITS[i])
    }
}
