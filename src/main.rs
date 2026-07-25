use clap::{Arg, Command};
use ff::Field;
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::elliptic_curve::BatchNormalize;
use k256::{AffinePoint, ProjectivePoint, Scalar};
use rand::thread_rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use ripemd::Ripemd160;
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// Full Base58 alphabet used by Bitcoin/VerusCoin addresses.
const BASE58_ALPHABET: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
// Characters intentionally excluded from Base58 due to visual ambiguity.
const INVALID_BASE58_CHARS: [char; 4] = ['0', 'O', 'I', 'l'];
// Size of the Base58 alphabet, used to approximate match probability.
const BASE58_ALPHABET_SIZE: f64 = 58.0;

// How many candidates a thread walks forward (via cheap EC point additions)
// from one random starting key before batch-normalizing them all to
// affine coordinates in a single shared modular inversion (Montgomery's
// batch-inversion trick). Normalizing one point at a time each pays its
// own inversion; batching amortizes that cost across the whole batch —
// measured ~2.5x faster than normalizing individually, on top of the
// point-addition-instead-of-full-scalar-mult optimization underneath it.
const BATCH_SIZE: usize = 512;

// How many keys a thread walks from one random starting point before
// picking a fresh random start again. Purely routine hygiene for very
// long runs — not a correctness or security requirement, since even at
// billions of keys/sec the walked range from one base is still a
// vanishingly small sliver of the full 2^256 keyspace.
const REBASE_INTERVAL: u64 = 200_000_000;

/// Validates a single pattern (prefix or suffix) against the Base58
/// alphabet. Returns Err with a human-readable, actionable message if the
/// pattern could never appear in a real Base58Check-encoded address.
fn validate_pattern(pattern: &str) -> Result<(), String> {
    let bad_chars: Vec<char> = pattern
        .chars()
        .filter(|c| INVALID_BASE58_CHARS.contains(c))
        .collect();

    if !bad_chars.is_empty() {
        let mut uniq = bad_chars.clone();
        uniq.dedup();
        let bad_list: String = uniq.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", ");
        return Err(format!(
            "'{}' can NEVER be found.\n  Invalid character(s): {}\n  \
             Base58 excludes 0 (zero), O (capital o), I (capital i), and l (lowercase L) \
             to avoid visual ambiguity.\n  Tip: use '1' instead of 'I' or 'O'; \
             lowercase 'i' or 'L' (capital) are valid substitutes.",
            pattern, bad_list
        ));
    }

    for c in pattern.chars() {
        if !BASE58_ALPHABET.contains(c) {
            return Err(format!(
                "'{}' contains '{}', which is not a valid Base58 character at all.",
                pattern, c
            ));
        }
    }

    Ok(())
}

/// Validates every pattern in a list; on any failure, print all problems
/// found and exit before any thread is spawned or any CPU time is burned
/// searching for something that is mathematically impossible to find.
fn validate_all_or_exit(patterns: &[String], label: &str) {
    let mut had_error = false;
    for pattern in patterns {
        if let Err(msg) = validate_pattern(pattern) {
            eprintln!("⚠️  Invalid {}: {}", label, msg);
            had_error = true;
        }
    }
    if had_error {
        eprintln!("\nAborting — fix the {}(es) above and try again.", label);
        std::process::exit(1);
    }
}

/// Every VerusCoin transparent address begins with 'R' — fixed by the
/// mainnet version byte, guaranteed no matter what the rest of the
/// address is. Since it's guaranteed anyway, a prefix that doesn't
/// already start with 'R' gets it prepended automatically rather than
/// making the user type a character that's never actually in question.
/// Returns the normalized prefix and whether it changed.
fn normalize_prefix(prefix: &str) -> (String, bool) {
    if prefix.starts_with('R') {
        (prefix.to_string(), false)
    } else {
        (format!("R{}", prefix), true)
    }
}

/// Reads a CLI value that may be a literal pattern or a path to a file of
/// patterns (one per line).
fn read_patterns(arg: &str) -> Vec<String> {
    if std::path::Path::new(arg).exists() {
        std::fs::read_to_string(arg)
            .unwrap()
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect()
    } else {
        vec![arg.to_string()]
    }
}

/// Finds the first prefix in `prefixes` that `addr` starts with.
/// Returns `Some(None)` if `prefixes` is empty (no constraint — always
/// satisfied), `Some(Some(p))` if a specific prefix `p` matched, or
/// `None` if a constraint exists but nothing matched.
fn match_any_prefix<'a>(prefixes: &'a [String], addr: &str) -> Option<Option<&'a str>> {
    if prefixes.is_empty() {
        return Some(None);
    }
    for p in prefixes {
        if addr.starts_with(p.as_str()) {
            return Some(Some(p.as_str()));
        }
    }
    None
}

/// Same as `match_any_prefix` but for suffixes (`ends_with`).
fn match_any_suffix<'a>(suffixes: &'a [String], addr: &str) -> Option<Option<&'a str>> {
    if suffixes.is_empty() {
        return Some(None);
    }
    for s in suffixes {
        if addr.ends_with(s.as_str()) {
            return Some(Some(s.as_str()));
        }
    }
    None
}

/// Estimates the expected number of addresses that must be generated
/// before at least one (prefix, suffix) combination matches, on average.
/// Prefixes count only characters beyond the guaranteed leading 'R';
/// suffixes have no guaranteed characters, so all of them count.
/// Combined prefix+suffix probability for a given pair is approximated as
/// the product of their individual probabilities (the standard
/// approximation used by vanity-address tools, treating leading- and
/// trailing-character windows as independent).
fn expected_tries(prefixes: &[String], suffixes: &[String]) -> f64 {
    let prefix_probs: Vec<f64> = if prefixes.is_empty() {
        vec![1.0]
    } else {
        prefixes
            .iter()
            .map(|p| BASE58_ALPHABET_SIZE.powi(-(p.len().saturating_sub(1) as i32)))
            .collect()
    };
    let suffix_probs: Vec<f64> = if suffixes.is_empty() {
        vec![1.0]
    } else {
        suffixes
            .iter()
            .map(|s| BASE58_ALPHABET_SIZE.powi(-(s.len() as i32)))
            .collect()
    };

    let mut total_probability = 0.0f64;
    for pp in &prefix_probs {
        for sp in &suffix_probs {
            total_probability += pp * sp;
        }
    }
    if total_probability <= 0.0 {
        f64::INFINITY
    } else {
        1.0 / total_probability
    }
}

/// Formats a duration in seconds as a human-readable string.
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

/// Probability of having found at least one match by now, under pure
/// randomness: P(at least one success in n tries) = 1 - (1-p)^n.
/// Computed via log for numerical stability with the very small p /
/// large n values typical here. Unlike ETA, this number only ever climbs
/// toward 100% as tries accumulate — it does NOT imply anything about the
/// odds of the next specific attempt, which are always exactly p no
/// matter how many tries came before (see the memoryless discussion on
/// the ETA calculation above).
fn cumulative_probability(p_per_try: f64, n_tries: u64) -> f64 {
    if p_per_try <= 0.0 {
        return 0.0;
    }
    if p_per_try >= 1.0 {
        return 1.0;
    }
    let log_survival = (n_tries as f64) * (1.0 - p_per_try).ln();
    1.0 - log_survival.exp()
}

/// Inverse of `cumulative_probability`: how many tries are needed to
/// reach a given cumulative success probability `q` (e.g. q=0.5 for the
/// point where a match is as likely as not, q=0.9 for a high-confidence
/// threshold). Solving 1-(1-p)^n = q for n gives n = ln(1-q)/ln(1-p).
/// This is still fundamentally a "from right now" estimate, not a
/// countdown — see the memorylessness note above; it doesn't shrink
/// just because earlier tries failed, only because the measured rate
/// changed or a match was found (which resets the baseline).
fn tries_for_probability(p_per_try: f64, q: f64) -> f64 {
    if p_per_try <= 0.0 {
        return f64::INFINITY;
    }
    if p_per_try >= 1.0 {
        return 1.0;
    }
    (1.0 - q).ln() / (1.0 - p_per_try).ln()
}

fn main() {
    let cpu_cores_str: &'static str = Box::leak(num_cpus::get().to_string().into_boxed_str());

    let matches = Command::new("verus-vanity")
        .version("0.3.0")
        .author("Your Name")
        .about("VerusCoin Vanity Wallet Generator")
        .arg(
            Arg::new("prefix")
                .short('p')
                .long("prefix")
                .help("Prefix string or filename with prefixes (one per line). 'R' is added automatically if omitted.")
                .num_args(1),
        )
        .arg(
            Arg::new("suffix")
                .short('s')
                .long("suffix")
                .help("Suffix string or filename with suffixes (one per line)")
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

    let prefix_arg = matches.get_one::<String>("prefix");
    let suffix_arg = matches.get_one::<String>("suffix");

    if prefix_arg.is_none() && suffix_arg.is_none() {
        eprintln!("⚠️  Provide at least one of --prefix/-p or --suffix/-s.");
        std::process::exit(1);
    }

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

    let raw_prefixes: Vec<String> = prefix_arg.map(|a| read_patterns(a)).unwrap_or_default();
    let suffixes: Vec<String> = suffix_arg.map(|a| read_patterns(a)).unwrap_or_default();

    // Every Verus address is guaranteed to start with 'R' — auto-prepend
    // it rather than requiring the user to type a character that was
    // never actually in question. See `normalize_prefix`.
    let mut any_normalized = false;
    let prefixes: Vec<String> = raw_prefixes
        .iter()
        .map(|p| {
            let (norm, changed) = normalize_prefix(p);
            if changed {
                any_normalized = true;
            }
            norm
        })
        .collect();

    validate_all_or_exit(&prefixes, "prefix");
    validate_all_or_exit(&suffixes, "suffix");

    println!("--- Starting verus-vanity ---");

    if !prefixes.is_empty() {
        if let Some(a) = prefix_arg {
            if std::path::Path::new(a).exists() {
                println!(
                    "Prefix file: {}",
                    std::fs::canonicalize(a).unwrap_or_else(|_| a.into()).display()
                );
            }
        }
        println!("Prefixes:");
        for (raw, norm) in raw_prefixes.iter().zip(prefixes.iter()) {
            if raw != norm {
                println!("  {}  (auto-prepended 'R' → {})", raw, norm);
            } else {
                println!("  {}", norm);
            }
        }
    } else {
        println!("Prefixes: none (matching by suffix only)");
    }

    if !suffixes.is_empty() {
        if let Some(a) = suffix_arg {
            if std::path::Path::new(a).exists() {
                println!(
                    "Suffix file: {}",
                    std::fs::canonicalize(a).unwrap_or_else(|_| a.into()).display()
                );
            }
        }
        println!("Suffixes:");
        for s in &suffixes {
            println!("  {}", s);
        }
    } else {
        println!("Suffixes: none (matching by prefix only)");
    }

    if any_normalized {
        println!("Note: every Verus address starts with 'R', so it was added automatically where missing.");
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
            std::fs::canonicalize(output).unwrap_or_else(|_| output.into()).display()
        );
    } else {
        println!("Output file: None");
    }

    let expected = expected_tries(&prefixes, &suffixes);
    println!("-----------------------------");

    println!("Starting with {} thread(s)...", threads);

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
    // Tracks the `keys_tried` value at the moment of the most recent
    // match, so the cumulative-probability stat can reset per match
    // instead of staying pinned near 100% for the rest of a multi-match
    // (-m N) run after early luck on the first couple of matches.
    let last_match_tries = Arc::new(AtomicU64::new(0));
    let start_time = Instant::now();

    let target_matches: f64 = if max_matches == -1 { 1.0 } else { max_matches.max(1) as f64 };

    {
        let keys_tried = Arc::clone(&keys_tried);
        let keys_tried_last = Arc::clone(&keys_tried_last);
        let found_count = Arc::clone(&found_count);
        let last_match_tries = Arc::clone(&last_match_tries);
        thread::spawn(move || {
            let mut printed_estimate = false;
            loop {
                thread::sleep(Duration::from_secs(1));
                let total = keys_tried.load(Ordering::Relaxed);
                let last = keys_tried_last.swap(total, Ordering::Relaxed);
                let rate = total.saturating_sub(last);

                let elapsed = start_time.elapsed().as_secs_f64();
                let average_rate = if elapsed > 0.0 { total as f64 / elapsed } else { 0.0 };

                let found_so_far = found_count.load(Ordering::Relaxed).max(0) as f64;
                let remaining_matches = (target_matches - found_so_far).max(0.0);
                let p_per_try = if expected.is_finite() && expected > 0.0 { 1.0 / expected } else { 0.0 };

                // Printed once, the first time a real rate is available —
                // a typical (median) time to find, not a countdown. It
                // won't reappear or shrink as the search runs; ongoing
                // progress is shown by the simple percentage below instead.
                if !printed_estimate && average_rate > 0.0 && remaining_matches > 0.0 {
                    let t50 = tries_for_probability(p_per_try, 0.5) * remaining_matches / average_rate;
                    println!("Estimated typical time to find: ~{}\n", format_duration(t50));
                    printed_estimate = true;
                }

                // Simple progress gauge: probability you'd already have a
                // match by now, under pure randomness. Always climbs
                // toward 100%, resets after each match on a -m N run.
                let progress = if remaining_matches > 0.0 {
                    let tries_since_last = total.saturating_sub(last_match_tries.load(Ordering::Relaxed));
                    format!("{:.1}%", cumulative_probability(p_per_try, tries_since_last) * 100.0)
                } else {
                    "done".to_string()
                };

                println!(
                    "Progress: {} ({} tried, {})",
                    progress,
                    format_with_si(total),
                    format_with_si_rate(rate)
                );
            }
        });
    }

    let mut handles = Vec::new();
    for _ in 0..threads {
        let prefixes = prefixes.clone();
        let suffixes = suffixes.clone();
        let output_writer = output_writer.clone();
        let found_count = found_count.clone();
        let keys_tried = keys_tried.clone();
        let last_match_tries = last_match_tries.clone();

        let handle = thread::spawn(move || {
            let mut rng_source = thread_rng();
            let mut rng = ChaCha20Rng::from_rng(&mut rng_source).expect("failed to seed RNG");

            // Start from one random scalar, then walk forward via cheap EC
            // point additions and batch-normalize in groups of BATCH_SIZE
            // (Montgomery's batch-inversion trick — one shared modular
            // inversion for the whole batch instead of one per point).
            // Cross-validated bit-identical against an independent
            // libsecp256k1 computation before shipping.
            let mut base_scalar = Scalar::random(&mut rng);
            let mut current = ProjectivePoint::GENERATOR * base_scalar;
            let mut offset: u64 = 0;
            let mut batch_buf: Vec<ProjectivePoint> = vec![ProjectivePoint::GENERATOR; BATCH_SIZE];

            'outer: loop {
                if max_matches != -1 && found_count.load(Ordering::Relaxed) >= max_matches {
                    break;
                }

                for slot in batch_buf.iter_mut() {
                    current += AffinePoint::GENERATOR;
                    *slot = current;
                }
                let affine_batch = ProjectivePoint::batch_normalize(batch_buf.as_slice());

                for (i, ap) in affine_batch.iter().enumerate() {
                    if max_matches != -1 && found_count.load(Ordering::Relaxed) >= max_matches {
                        break 'outer;
                    }

                    let this_offset = offset + 1 + i as u64;
                    let compressed = ap.to_encoded_point(true);
                    let addr = public_key_to_address(compressed.as_bytes(), 0x3c);

                    if let (Some(p_opt), Some(s_opt)) =
                        (match_any_prefix(&prefixes, &addr), match_any_suffix(&suffixes, &addr))
                    {
                        let sk_scalar = base_scalar + Scalar::from(this_offset);
                        let sk_bytes: [u8; 32] = sk_scalar.to_bytes().into();
                        let wif = private_key_to_wif(&sk_bytes, 0xbc, true); // VerusCoin WIF prefix
                        let priv_hex = hex::encode(sk_bytes);

                        let desc = match (p_opt, s_opt) {
                            (Some(p), Some(s)) => format!("prefix '{}' + suffix '{}'", p, s),
                            (Some(p), None) => format!("prefix '{}'", p),
                            (None, Some(s)) => format!("suffix '{}'", s),
                            (None, None) => "match".to_string(),
                        };

                        let found_num = found_count.fetch_add(1, Ordering::Relaxed) + 1;
                        // Reset the "odds you'd have found it by now" stat
                        // to start counting fresh from this match forward.
                        // fetch_max (rather than a plain store) keeps this
                        // correct even if two threads find matches close
                        // together and race here.
                        last_match_tries.fetch_max(keys_tried.load(Ordering::Relaxed), Ordering::Relaxed);

                        println!("----- MATCH {} for {} FOUND -----", found_num, desc);
                        println!("Address: {}", addr);
                        println!("WIF: {}", wif);
                        println!("Private Key (hex): {}", priv_hex);
                        println!("Scan this QR code to import the WIF into your wallet app:");
                        println!("{}", wif_to_qr_string(&wif));
                        println!("-------------------------\n");

                        if let Some(ref output_mutex) = output_writer {
                            let mut output = output_mutex.lock().unwrap();
                            writeln!(output, "----- MATCH {} for {} FOUND -----", found_num, desc).ok();
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

                offset += BATCH_SIZE as u64;
                keys_tried.fetch_add(BATCH_SIZE as u64, Ordering::Relaxed);

                if offset >= REBASE_INTERVAL {
                    base_scalar = Scalar::random(&mut rng);
                    current = ProjectivePoint::GENERATOR * base_scalar;
                    offset = 0;
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

fn public_key_to_address(pubkey_compressed: &[u8], version: u8) -> String {
    let sha256_hash = Sha256::digest(pubkey_compressed);
    let ripemd_hash = Ripemd160::digest(sha256_hash);

    let mut addr_bytes = [0u8; 25];
    addr_bytes[0] = version;
    addr_bytes[1..21].copy_from_slice(&ripemd_hash);

    let checksum_full = Sha256::digest(Sha256::digest(&addr_bytes[0..21]));
    addr_bytes[21..25].copy_from_slice(&checksum_full[0..4]);

    bs58::encode(&addr_bytes[..]).into_string()
}

/// Renders a scannable QR code of the WIF directly to the terminal using
/// Unicode block characters, so it can be imported via a wallet's
/// "scan QR code" option instead of typing it in by hand.
fn wif_to_qr_string(wif: &str) -> String {
    match qrcode::QrCode::new(wif.as_bytes()) {
        Ok(code) => code
            .render::<qrcode::render::unicode::Dense1x2>()
            .quiet_zone(true)
            .build(),
        Err(e) => format!("(failed to generate QR code: {})", e),
    }
}

fn private_key_to_wif(sk_bytes: &[u8; 32], version_byte: u8, compressed: bool) -> String {
    let mut bytes = [0u8; 34];
    bytes[0] = version_byte;
    bytes[1..33].copy_from_slice(sk_bytes);
    let payload_len = if compressed {
        bytes[33] = 0x01;
        34
    } else {
        33
    };

    let checksum_full = Sha256::digest(Sha256::digest(&bytes[0..payload_len]));

    let mut extended = [0u8; 38];
    extended[0..payload_len].copy_from_slice(&bytes[0..payload_len]);
    extended[payload_len..payload_len + 4].copy_from_slice(&checksum_full[0..4]);

    bs58::encode(&extended[0..payload_len + 4]).into_string()
}

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
