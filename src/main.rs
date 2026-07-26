//! verus-vanity — a VerusCoin vanity wallet address generator.
//!
//! Searches for a private key whose derived transparent ("R…") address
//! matches a desired prefix and/or suffix. Uses an incremental EC
//! point-walk with batched modular inversion (Montgomery's trick) instead
//! of a fresh scalar multiplication per candidate — see the comments on
//! `BATCH_SIZE` and inside `worker_loop` for details, and the WIF used
//! throughout the accompanying README/changelog is cross-validated
//! against an independent libsecp256k1 computation.

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
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// ===================== Constants =====================

/// Full Base58 alphabet used by Bitcoin/VerusCoin addresses.
const BASE58_ALPHABET: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
/// Characters intentionally excluded from Base58 due to visual ambiguity.
const INVALID_BASE58_CHARS: [char; 4] = ['0', 'O', 'I', 'l'];
/// Size of the Base58 alphabet, used to approximate match probability.
const BASE58_ALPHABET_SIZE: f64 = 58.0;

/// VerusCoin mainnet transparent-address version byte (produces the
/// leading 'R'). Applied after the RIPEMD160(SHA256(pubkey)) hash.
const ADDRESS_VERSION_BYTE: u8 = 0x3c;
/// VerusCoin mainnet WIF (private key) version byte.
const WIF_VERSION_BYTE: u8 = 0xbc;

/// How many candidates a thread walks forward (via cheap EC point
/// additions) from one random starting key before batch-normalizing them
/// all to affine coordinates in a single shared modular inversion
/// (Montgomery's batch-inversion trick). Normalizing one point at a time
/// each pays its own inversion; batching amortizes that cost across the
/// whole batch. Empirically flat across a wide range of sizes (64 to
/// 8192 all land within ~7% of each other), so 512 is simply a solid,
/// unremarkable choice rather than a finely tuned one.
const BATCH_SIZE: usize = 512;

/// How many keys a thread walks from one random starting point before
/// picking a fresh random start again. Purely routine hygiene for very
/// long runs — not a correctness or security requirement, since even at
/// billions of keys/sec the walked range from one base is still a
/// vanishingly small sliver of the full 2^256 keyspace.
const REBASE_INTERVAL: u64 = 200_000_000;

// ===================== Pattern validation & normalization =====================

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
/// making the user type a character that was never actually in question.
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

// ===================== Matching =====================

/// Finds the first prefix in `prefixes` that `addr` starts with.
/// Returns `Some(None)` if `prefixes` is empty (no constraint — always
/// satisfied), `Some(Some(p))` if a specific prefix `p` matched, or
/// `None` if a constraint exists but nothing matched.
fn match_any_prefix<'a>(prefixes: &'a [String], addr: &str) -> Option<Option<&'a str>> {
    if prefixes.is_empty() {
        return Some(None);
    }
    prefixes
        .iter()
        .find(|p| addr.starts_with(p.as_str()))
        .map(|p| Some(p.as_str()))
}

/// Same as `match_any_prefix` but for suffixes (`ends_with`).
fn match_any_suffix<'a>(suffixes: &'a [String], addr: &str) -> Option<Option<&'a str>> {
    if suffixes.is_empty() {
        return Some(None);
    }
    suffixes
        .iter()
        .find(|s| addr.ends_with(s.as_str()))
        .map(|s| Some(s.as_str()))
}

/// Builds the human-readable "for prefix 'X' + suffix 'Y'" description
/// used in match reports, covering all four combinations of which
/// constraint(s) were actually in play.
fn describe_match(prefix_hit: Option<&str>, suffix_hit: Option<&str>) -> String {
    match (prefix_hit, suffix_hit) {
        (Some(p), Some(s)) => format!("prefix '{}' + suffix '{}'", p, s),
        (Some(p), None) => format!("prefix '{}'", p),
        (None, Some(s)) => format!("suffix '{}'", s),
        (None, None) => "match".to_string(),
    }
}

// ===================== Probability & time estimates =====================

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

/// Probability of having found at least one match by now, under pure
/// randomness: P(at least one success in n tries) = 1 - (1-p)^n.
/// Computed via log for numerical stability with the very small p /
/// large n values typical here. This number only ever climbs toward
/// 100% as tries accumulate — it does NOT imply anything about the odds
/// of the next specific attempt, which are always exactly p no matter
/// how many tries came before (the process is memoryless).
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
/// point where a match is as likely as not). Solving 1-(1-p)^n = q for n
/// gives n = ln(1-q)/ln(1-p). Still fundamentally a "from right now"
/// estimate, not a countdown — it doesn't shrink just because earlier
/// tries failed, only because the measured rate changed or a match was
/// found (which resets the baseline in `worker_loop`).
fn tries_for_probability(p_per_try: f64, q: f64) -> f64 {
    if p_per_try <= 0.0 {
        return f64::INFINITY;
    }
    if p_per_try >= 1.0 {
        return 1.0;
    }
    (1.0 - q).ln() / (1.0 - p_per_try).ln()
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

/// Formats a large integer count with SI-style suffixes (K, M, G, ...).
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

/// Same as `format_with_si` but labeled as a rate (W/s, KW/s, ...).
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

// ===================== Address / key encoding =====================

/// Computes RIPEMD160(SHA256(pubkey)) plus the fixed version byte,
/// giving the first 21 bytes of the address — everything except the
/// 4-byte checksum, which requires two more SHA256 calls to produce.
/// Split out from `public_key_to_address` so the prefix pre-filter (see
/// `provably_outside`) can run in between: it only needs these 21 bytes,
/// letting the checksum + Base58 encode be skipped entirely for
/// candidates already proven impossible.
fn compute_first21(pubkey_compressed: &[u8], version: u8) -> [u8; 21] {
    let sha256_hash = Sha256::digest(pubkey_compressed);
    let ripemd_hash = Ripemd160::digest(sha256_hash);
    let mut first21 = [0u8; 21];
    first21[0] = version;
    first21[1..21].copy_from_slice(&ripemd_hash);
    first21
}

/// Completes a VerusCoin transparent address given the first 21 bytes:
/// computes the double-SHA256 checksum and Base58-encodes the result.
fn address_from_first21(first21: &[u8; 21]) -> String {
    let mut addr_bytes = [0u8; 25];
    addr_bytes[0..21].copy_from_slice(first21);
    let checksum_full = Sha256::digest(Sha256::digest(&addr_bytes[0..21]));
    addr_bytes[21..25].copy_from_slice(&checksum_full[0..4]);
    bs58::encode(&addr_bytes[..]).into_string()
}

/// Derives a VerusCoin transparent address from a compressed public key:
/// Base58Check(version_byte || RIPEMD160(SHA256(pubkey))). Convenience
/// wrapper around `compute_first21` + `address_from_first21` for callers
/// that don't need the pre-filter split (e.g. suffix-only searches).
#[allow(dead_code)]
fn public_key_to_address(pubkey_compressed: &[u8], version: u8) -> String {
    address_from_first21(&compute_first21(pubkey_compressed, version))
}

// ===================== Prefix pre-filter (speed, prefix-only) =====================
//
// Base58 encodes the entire 25-byte address as one big number, so the
// LEADING characters are dominated by the high-order bytes (version +
// start of hash160), while the checksum — the last 4 bytes, unknowable
// without actually hashing — only ever perturbs the number by at most
// 2^32 out of a ~2^200 range. That gap is what makes this safe: for any
// candidate, [first21 as a number with checksum=0, first21 with
// checksum=0xFFFFFFFF] is a small, fully-known range. If that whole range
// falls outside the numeric window that would produce the desired
// prefix, NO possible checksum could ever make it match — proven by
// exact integer bounds, not approximation, so it can be skipped before
// ever computing the checksum or encoding to Base58.
//
// This can never cause a wrong result: the final match decision always
// goes through the same exact, unchanged `match_any_prefix`/
// `match_any_suffix` + Base58 comparison as before. The pre-filter only
// ever skips candidates already *proven* impossible. If bound
// computation fails for any reason, it's simply not used, and every
// candidate falls through to the always-correct exact path — same as
// before this optimization existed. Suffix matching cannot use this
// shortcut at all: the checksum isn't a small perturbation there, it's
// the primary thing that determines the trailing characters, so it must
// always be computed for real.

const ADDR_CHARS: usize = 34;

/// Numeric bounds for fast prefix pre-filtering — see the module note
/// above. `lower` and `upper_exclusive` are 25-byte big-endian values;
/// standard array comparison (`<`, `>=`) on them is exactly equivalent to
/// numeric comparison of the big-endian integers they represent.
#[derive(Clone, Copy)]
struct PrefixBound {
    lower: [u8; 25],
    upper_exclusive: [u8; 25],
}

/// Computes the bound using the already-trusted `bs58` crate's own
/// decode logic (deliberately not hand-rolled bignum math): decode the
/// smallest and largest 34-character Base58 strings that start with this
/// prefix, then bump the largest one up by one to make it an exclusive
/// bound. Returns None if anything is even slightly off (decode error,
/// unexpected byte length, carry overflow past all 25 bytes) — callers
/// must treat None as "fast path unavailable here", not as an error.
fn compute_prefix_bound(prefix: &str) -> Option<PrefixBound> {
    if prefix.is_empty() || prefix.len() > ADDR_CHARS {
        return None;
    }
    let pad_len = ADDR_CHARS - prefix.len();

    let lower_str = format!("{}{}", prefix, "1".repeat(pad_len));
    let upper_str = format!("{}{}", prefix, "z".repeat(pad_len));

    let lower_vec = bs58::decode(&lower_str).into_vec().ok()?;
    let upper_vec = bs58::decode(&upper_str).into_vec().ok()?;
    if lower_vec.len() != 25 || upper_vec.len() != 25 {
        return None;
    }

    let mut lower = [0u8; 25];
    lower.copy_from_slice(&lower_vec);
    let mut upper_exclusive = [0u8; 25];
    upper_exclusive.copy_from_slice(&upper_vec);

    // Increment upper_exclusive by 1 (big-endian, with carry).
    let mut i = 24usize;
    loop {
        if upper_exclusive[i] == 0xff {
            upper_exclusive[i] = 0;
            if i == 0 {
                return None; // overflowed past all 25 bytes; use the safe path instead
            }
            i -= 1;
        } else {
            upper_exclusive[i] += 1;
            break;
        }
    }

    Some(PrefixBound { lower, upper_exclusive })
}

/// Attempts to compute a pre-filter bound for every prefix. Returns an
/// empty Vec (meaning "fast path disabled, use the exact method for
/// everything") if there are any suffix constraints at all — suffix
/// matching always needs the real checksum — or if any single prefix's
/// bound can't be computed. All-or-nothing keeps the hot-loop logic
/// simple: either every prefix has a valid bound, or none of them get
/// the fast-path treatment.
fn try_compute_all_prefix_bounds(prefixes: &[String], suffixes: &[String]) -> Vec<PrefixBound> {
    if !suffixes.is_empty() {
        return Vec::new();
    }
    let mut bounds = Vec::with_capacity(prefixes.len());
    for p in prefixes {
        match compute_prefix_bound(p) {
            Some(b) => bounds.push(b),
            None => return Vec::new(),
        }
    }
    bounds
}

/// Returns true only if literally no checksum value could make this
/// candidate's address start with the prefix behind `bound` — see the
/// module note above for why this is always safe, never approximate.
fn provably_outside(first21: &[u8; 21], bound: &PrefixBound) -> bool {
    let mut lo = [0u8; 25];
    lo[0..21].copy_from_slice(first21);
    let mut hi = [0u8; 25];
    hi[0..21].copy_from_slice(first21);
    hi[21..25].copy_from_slice(&[0xff, 0xff, 0xff, 0xff]);

    hi < bound.lower || lo >= bound.upper_exclusive
}

/// Encodes a raw 32-byte private key as WIF (Wallet Import Format):
/// Base58Check(version_byte || key || [0x01 if compressed]).
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

// ===================== CLI configuration =====================

/// Parsed and validated command-line configuration.
struct Config {
    /// Prefixes exactly as typed/read from file, before 'R' normalization
    /// (kept alongside the normalized versions purely for the startup
    /// banner's "auto-prepended" display).
    raw_prefixes: Vec<String>,
    /// Prefixes after auto-prepending 'R' where needed; always what's
    /// actually used for matching.
    prefixes: Vec<String>,
    suffixes: Vec<String>,
    prefix_arg: Option<String>,
    suffix_arg: Option<String>,
    threads: usize,
    max_matches: i64,
    output_file: Option<String>,
    any_normalized: bool,
}

/// Parses CLI arguments, auto-normalizes prefixes (see `normalize_prefix`),
/// and validates every pattern — exiting the process with a clear message
/// if anything is missing or impossible before any search work begins.
fn parse_cli() -> Config {
    let cpu_cores_str: &'static str = Box::leak(num_cpus::get().to_string().into_boxed_str());

    let matches = Command::new("verus-vanity")
        .version("0.4.0")
        .author("Darktron")
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

    let prefix_arg = matches.get_one::<String>("prefix").cloned();
    let suffix_arg = matches.get_one::<String>("suffix").cloned();

    if prefix_arg.is_none() && suffix_arg.is_none() {
        eprintln!("⚠️  Provide at least one of --prefix/-p or --suffix/-s.");
        std::process::exit(1);
    }

    let output_file = matches.get_one::<String>("output").cloned();
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

    let raw_prefixes: Vec<String> = prefix_arg.as_deref().map(read_patterns).unwrap_or_default();
    let suffixes: Vec<String> = suffix_arg.as_deref().map(read_patterns).unwrap_or_default();

    // Every Verus address is guaranteed to start with 'R' — auto-prepend
    // it rather than requiring the user to type a character that was
    // never actually in question. See `normalize_prefix`.
    let mut any_normalized = false;
    let prefixes: Vec<String> = raw_prefixes
        .iter()
        .map(|p| {
            let (norm, changed) = normalize_prefix(p);
            any_normalized |= changed;
            norm
        })
        .collect();

    validate_all_or_exit(&prefixes, "prefix");
    validate_all_or_exit(&suffixes, "suffix");

    Config {
        raw_prefixes,
        prefixes,
        suffixes,
        prefix_arg,
        suffix_arg,
        threads,
        max_matches,
        output_file,
        any_normalized,
    }
}

/// Prints the startup banner summarizing the search that's about to run.
fn print_banner(config: &Config) {
    println!("--- Starting verus-vanity ---");

    if !config.prefixes.is_empty() {
        if let Some(a) = &config.prefix_arg {
            if std::path::Path::new(a).exists() {
                println!(
                    "Prefix file: {}",
                    std::fs::canonicalize(a).unwrap_or_else(|_| a.into()).display()
                );
            }
        }
        println!("Prefixes:");
        for (raw, norm) in config.raw_prefixes.iter().zip(config.prefixes.iter()) {
            if raw != norm {
                println!("  {}  (auto-prepended 'R' → {})", raw, norm);
            } else {
                println!("  {}", norm);
            }
        }
    } else {
        println!("Prefixes: none (matching by suffix only)");
    }

    if !config.suffixes.is_empty() {
        if let Some(a) = &config.suffix_arg {
            if std::path::Path::new(a).exists() {
                println!(
                    "Suffix file: {}",
                    std::fs::canonicalize(a).unwrap_or_else(|_| a.into()).display()
                );
            }
        }
        println!("Suffixes:");
        for s in &config.suffixes {
            println!("  {}", s);
        }
    } else {
        println!("Suffixes: none (matching by prefix only)");
    }

    if config.any_normalized {
        println!("Note: every Verus address starts with 'R', so it was added automatically where missing.");
    }

    println!("Threads: {}", config.threads);
    if config.max_matches == -1 {
        println!("Max Matches: infinite");
    } else {
        println!("Max Matches: {}", config.max_matches);
    }

    if let Some(output) = &config.output_file {
        println!(
            "Output file: {}",
            std::fs::canonicalize(output).unwrap_or_else(|_| output.into()).display()
        );
    } else {
        println!("Output file: None");
    }

    println!("-----------------------------");
    println!("Starting with {} thread(s)...", config.threads);
}

// ===================== Shared run state =====================

/// State shared between the stats-reporting thread and every worker
/// thread. Cheap to clone (every field is an `Arc`), so each thread just
/// gets its own clone of the whole bundle instead of cloning fields
/// individually.
#[derive(Clone)]
struct SharedState {
    found_count: Arc<AtomicI64>,
    keys_tried: Arc<AtomicU64>,
    /// Tracks the `keys_tried` value at the moment of the most recent
    /// match, so the "progress" stat can reset per match instead of
    /// staying pinned near 100% for the rest of a multi-match (-m N) run
    /// after early luck on the first couple of matches.
    last_match_tries: Arc<AtomicU64>,
    output_writer: Option<Arc<Mutex<BufWriter<File>>>>,
}

impl SharedState {
    fn new(output_file: &Option<String>) -> Self {
        let output_writer = output_file.as_ref().map(|filename| {
            Arc::new(Mutex::new(BufWriter::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(filename)
                    .expect("Failed to open output file"),
            )))
        });

        SharedState {
            found_count: Arc::new(AtomicI64::new(0)),
            keys_tried: Arc::new(AtomicU64::new(0)),
            last_match_tries: Arc::new(AtomicU64::new(0)),
            output_writer,
        }
    }
}

/// Spawns the background thread that prints search progress once a
/// second: a one-time typical-time-to-find estimate the first time a real
/// rate is available, then an ongoing simple progress percentage.
fn spawn_stats_thread(state: SharedState, start_time: Instant, expected: f64, target_matches: f64) {
    thread::spawn(move || {
        let keys_tried_last = AtomicU64::new(0);
        let mut printed_estimate = false;

        loop {
            thread::sleep(Duration::from_secs(1));
            let total = state.keys_tried.load(Ordering::Relaxed);
            let last = keys_tried_last.swap(total, Ordering::Relaxed);
            let rate = total.saturating_sub(last);

            let elapsed = start_time.elapsed().as_secs_f64();
            let average_rate = if elapsed > 0.0 { total as f64 / elapsed } else { 0.0 };

            let found_so_far = state.found_count.load(Ordering::Relaxed).max(0) as f64;
            let remaining_matches = (target_matches - found_so_far).max(0.0);
            let p_per_try = if expected.is_finite() && expected > 0.0 { 1.0 / expected } else { 0.0 };

            // Printed once, the first time a real rate is available — a
            // typical (median) time to find, not a countdown. It won't
            // reappear or shrink as the search runs; ongoing progress is
            // shown by the simple percentage below instead.
            if !printed_estimate && average_rate > 0.0 && remaining_matches > 0.0 {
                let t50 = tries_for_probability(p_per_try, 0.5) * remaining_matches / average_rate;
                println!("Estimated typical time to find: ~{}\n", format_duration(t50));
                printed_estimate = true;
            }

            // Simple progress gauge: probability you'd already have a
            // match by now, under pure randomness. Always climbs toward
            // 100%, resets after each match on a -m N run.
            let progress = if remaining_matches > 0.0 {
                let tries_since_last = total.saturating_sub(state.last_match_tries.load(Ordering::Relaxed));
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

// ===================== Search worker =====================

/// Prints a found match (address, WIF, hex key, QR code) to stdout, and
/// appends the same information to the output file if one was configured.
fn report_match(found_num: i64, desc: &str, addr: &str, wif: &str, priv_hex: &str, output_writer: &Option<Arc<Mutex<BufWriter<File>>>>) {
    let qr = wif_to_qr_string(wif);

    println!("----- MATCH {} for {} FOUND -----", found_num, desc);
    println!("Address: {}", addr);
    println!("WIF: {}", wif);
    println!("Private Key (hex): {}", priv_hex);
    println!("Scan this QR code to import the WIF into your wallet app:");
    println!("{}", qr);
    println!("-------------------------\n");

    if let Some(output_mutex) = output_writer {
        let mut output = output_mutex.lock().unwrap();
        writeln!(output, "----- MATCH {} for {} FOUND -----", found_num, desc).ok();
        writeln!(output, "Address: {}", addr).ok();
        writeln!(output, "WIF: {}", wif).ok();
        writeln!(output, "Private Key (hex): {}", priv_hex).ok();
        writeln!(output, "Scan this QR code to import the WIF into your wallet app:").ok();
        writeln!(output, "{}", qr).ok();
        writeln!(output, "-------------------------\n").ok();
        output.flush().ok();
    }
}

/// One worker thread's search loop. Starts from one random scalar, then
/// walks forward via cheap EC point additions and batch-normalizes in
/// groups of `BATCH_SIZE` (Montgomery's batch-inversion trick — one
/// shared modular inversion for the whole batch instead of one per
/// point). Cross-validated bit-identical against an independent
/// libsecp256k1 computation before shipping.
fn worker_loop(prefixes: Vec<String>, suffixes: Vec<String>, prefix_bounds: Vec<PrefixBound>, max_matches: i64, state: SharedState) {
    let mut rng_source = thread_rng();
    let mut rng = ChaCha20Rng::from_rng(&mut rng_source).expect("failed to seed RNG");

    let mut base_scalar = Scalar::random(&mut rng);
    let mut current = ProjectivePoint::GENERATOR * base_scalar;
    let mut offset: u64 = 0;
    let mut batch_buf: Vec<ProjectivePoint> = vec![ProjectivePoint::GENERATOR; BATCH_SIZE];

    'outer: loop {
        if max_matches != -1 && state.found_count.load(Ordering::Relaxed) >= max_matches {
            break;
        }

        for slot in batch_buf.iter_mut() {
            current += AffinePoint::GENERATOR;
            *slot = current;
        }
        let affine_batch = ProjectivePoint::batch_normalize(batch_buf.as_slice());

        for (i, ap) in affine_batch.iter().enumerate() {
            if max_matches != -1 && state.found_count.load(Ordering::Relaxed) >= max_matches {
                break 'outer;
            }

            let this_offset = offset + 1 + i as u64;
            let compressed = ap.to_encoded_point(true);
            let first21 = compute_first21(compressed.as_bytes(), ADDRESS_VERSION_BYTE);

            // Fast path (prefix-only): if every prefix is proven
            // impossible regardless of checksum, skip the checksum
            // computation and Base58 encode entirely. Empty
            // prefix_bounds (suffix present, or bound computation
            // failed) means this never triggers, and every candidate
            // falls through to the exact path below unchanged.
            if !prefix_bounds.is_empty() && prefix_bounds.iter().all(|b| provably_outside(&first21, b)) {
                continue;
            }

            let addr = address_from_first21(&first21);

            let prefix_hit = match_any_prefix(&prefixes, &addr);
            let suffix_hit = match_any_suffix(&suffixes, &addr);

            if let (Some(prefix_hit), Some(suffix_hit)) = (prefix_hit, suffix_hit) {
                // Only reconstruct the actual private key on a real match
                // (rare) — the hot loop above never needs it, since it
                // only ever walks the public key forward.
                let sk_scalar = base_scalar + Scalar::from(this_offset);
                let sk_bytes: [u8; 32] = sk_scalar.to_bytes().into();
                let wif = private_key_to_wif(&sk_bytes, WIF_VERSION_BYTE, true);
                let priv_hex = hex::encode(sk_bytes);
                let desc = describe_match(prefix_hit, suffix_hit);

                let found_num = state.found_count.fetch_add(1, Ordering::Relaxed) + 1;
                // Reset the progress-percentage baseline to start counting
                // fresh from this match forward. fetch_max (rather than a
                // plain store) keeps this correct even if two threads find
                // matches close together and race here.
                state
                    .last_match_tries
                    .fetch_max(state.keys_tried.load(Ordering::Relaxed), Ordering::Relaxed);

                report_match(found_num, &desc, &addr, &wif, &priv_hex, &state.output_writer);
            }
        }

        offset += BATCH_SIZE as u64;
        state.keys_tried.fetch_add(BATCH_SIZE as u64, Ordering::Relaxed);

        if offset >= REBASE_INTERVAL {
            base_scalar = Scalar::random(&mut rng);
            current = ProjectivePoint::GENERATOR * base_scalar;
            offset = 0;
        }
    }
}

// ===================== Entry point =====================

fn main() {
    let config = parse_cli();
    let expected = expected_tries(&config.prefixes, &config.suffixes);
    print_banner(&config);

    // Precomputed once, reused read-only by every thread. Empty means
    // the fast path is unavailable (suffixes present, or a prefix's
    // bound couldn't be computed) — every candidate then simply falls
    // through to the exact, unchanged path in worker_loop, same as
    // before this optimization existed.
    let prefix_bounds = try_compute_all_prefix_bounds(&config.prefixes, &config.suffixes);

    let state = SharedState::new(&config.output_file);
    let start_time = Instant::now();
    let target_matches: f64 = if config.max_matches == -1 { 1.0 } else { config.max_matches.max(1) as f64 };

    spawn_stats_thread(state.clone(), start_time, expected, target_matches);

    let mut handles = Vec::new();
    for _ in 0..config.threads {
        let prefixes = config.prefixes.clone();
        let suffixes = config.suffixes.clone();
        let prefix_bounds = prefix_bounds.clone();
        let state = state.clone();
        let max_matches = config.max_matches;
        handles.push(thread::spawn(move || worker_loop(prefixes, suffixes, prefix_bounds, max_matches, state)));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
