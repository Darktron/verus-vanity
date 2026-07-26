//! verus-vanity — a VerusCoin vanity wallet address generator.
//!
//! Searches for a private key whose derived transparent ("R…") address
//! matches a desired prefix, infix, and/or suffix.
//!
//! Candidates are produced in batches of independent affine point
//! additions from a precomputed table, sharing one modular inversion per
//! batch (Montgomery's trick) — see "Batched affine point generation"
//! below. Most candidates are then discarded by numeric pre-filters
//! before their checksum is ever computed (see "Prefix & suffix
//! pre-filters"). Every reported match is re-derived from its private
//! key through an independent code path and checked before being
//! printed, so a key that would not control its address is never
//! emitted.

use clap::{Arg, ArgAction, Command};
use ff::Field;
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::{AffinePoint, FieldElement, ProjectivePoint, Scalar};
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

/// How many candidate keys are produced per batch — see the
/// "Batched affine point generation" section below for what a batch
/// actually does. The single modular inversion each batch requires is
/// amortized across this many points, so bigger is better up to a point,
/// but the per-point cost is dominated by multiplications that scale
/// linearly either way.
///
/// Re-measured empirically for the current algorithm: 256 is slightly
/// slower (~2.03 MW/s), while 512 and 1024 are equivalent (~2.1-2.2
/// MW/s). 512 is kept because it gets the same throughput as 1024 for
/// half the memory.
///
/// IMPORTANT: this controls STACK usage. Two arrays of this length live
/// on each worker thread's stack (`denominators` and `invert_scratch`),
/// so 512 uses roughly 64KB of a thread's default 2MB stack. Raising it
/// substantially (past ~8192) risks a stack overflow and buys nothing
/// measurable given the flat scaling above.
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
    // Every VerusCoin address is exactly ADDR_CHARS long (the fixed
    // version byte pins the encoded value into a range that always
    // produces that many Base58 digits), so anything longer than that
    // cannot occur anywhere in one. Without this check such a pattern is
    // accepted and searched for forever.
    let char_count = pattern.chars().count();
    if char_count > ADDR_CHARS {
        return Err(format!(
            "'{}' can NEVER be found.\n  It is {} characters long, but every VerusCoin address \
             is exactly {} characters, so no address can contain it.",
            pattern, char_count, ADDR_CHARS
        ));
    }
    if char_count == 0 {
        return Err(
            "an empty pattern was given.\n  Provide at least one character to search for."
                .to_string(),
        );
    }

    let mut bad_chars: Vec<char> = pattern
        .chars()
        .filter(|c| INVALID_BASE58_CHARS.contains(c))
        .collect();

    if !bad_chars.is_empty() {
        // Sort before dedup: dedup only removes *consecutive* duplicates,
        // so without sorting a pattern like "RIOI" would report "I, O, I".
        bad_chars.sort_unstable();
        bad_chars.dedup();
        let bad_list: String = bad_chars.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", ");
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

/// Characters that can appear as the SECOND character of a VerusCoin
/// address (position 1, right after the guaranteed leading 'R') — kept
/// as a named constant purely so error messages can give a precise,
/// simple explanation for the common case. Used only for MESSAGING; the
/// actual pass/fail decision always comes from `validate_prefix_achievable`
/// below, which is provably correct at any depth (see its own doc
/// comment for why the simple 24-character rule isn't the whole story).
///
/// Why 24 and not 58: N (the 25-byte address value: fixed version byte
/// 0x3C, followed by 192 bits of hash160+checksum that vary freely)
/// always falls in the range [0x3C * 2^192, 0x3D * 2^192) — narrower
/// than one full second-character Base58 bucket's worth of possibilities
/// can spread across (58^32 vs the 2^192 span), so it only ever touches
/// 24 of the 58 possible second-character buckets. Confirmed by exact
/// interval arithmetic and empirically against 4,231 real unbiased
/// generated addresses: exactly these 24 characters appeared, zero
/// exceptions.
const VALID_SECOND_CHARS: &str = "9ABCDEFGHJKLMNPQRSTUVWXY";

/// Validates that a prefix is actually reachable by some real VerusCoin
/// address — not just "does it use valid Base58 characters", but "does
/// ANY possible hash160+checksum combination actually produce this
/// exact prefix".
///
/// The naive expectation is that only the very first character is
/// constrained (fixed to 'R') and everything after is freely chosen.
/// That's wrong in a way that goes deeper than it first appears: the
/// second character turns out to be restricted to 24 specific values
/// (see `VALID_SECOND_CHARS`) — and *within* the two boundary values of
/// that restriction, the THIRD character is itself further restricted,
/// recursively, for as many characters as the boundary keeps getting
/// touched. Concretely: 'R9A' is unreachable even though '9' and 'A' are
/// each independently valid at their positions, and 'R9HA' is *also*
/// unreachable (while 'R9Ha' — lowercase — is fine). Hand-deriving each
/// layer of this is exactly the kind of thing that's easy to get subtly
/// wrong or leave incomplete (as happened here — an earlier version of
/// this validation only caught the second-character case and missed
/// deeper combinations like 'R9HA').
///
/// Rather than deriving the recursive boundary structure by hand, this
/// reuses the exact same machinery already proven correct for the speed
/// pre-filter: `compute_prefix_bound` computes the prefix's own Base58
/// numeric bucket (the same bucket used to safely skip impossible
/// candidates during the search), and this just checks whether that
/// bucket overlaps AT ALL with the true achievable range for a VerusCoin
/// address (version byte 0x3C followed by every possible 24-byte
/// hash160+checksum combination). If there's no overlap, no possible
/// private key could ever produce this prefix — proven by the same exact
/// interval arithmetic as the pre-filter, not by re-deriving boundary
/// rules by hand. Verified to correctly reproduce every case checked by
/// hand above, including the deeper 'R9HA' one.
fn validate_prefix_achievable(prefix: &str) -> Result<(), String> {
    let bound = match compute_prefix_bound(prefix) {
        Some(b) => b,
        None => return Ok(()), // inconclusive (e.g. prefix too long) — don't block on it
    };

    let mut achievable_low = [0u8; 25];
    achievable_low[0] = ADDRESS_VERSION_BYTE;
    let mut achievable_high = [0u8; 25];
    achievable_high[0] = ADDRESS_VERSION_BYTE + 1;

    let overlaps = !(bound.upper_exclusive <= achievable_low || bound.lower >= achievable_high);
    if overlaps {
        return Ok(());
    }

    // Give the most specific explanation available; the pass/fail
    // decision above is already final regardless of which message fires.
    let chars: Vec<char> = prefix.chars().collect();
    if chars.len() >= 2 && !VALID_SECOND_CHARS.contains(chars[1]) {
        return Err(format!(
            "'{}' can NEVER be found.\n  The second character '{}' is impossible: VerusCoin's \
             fixed version byte restricts an address's second character to only these {} values: \
             {}\n  This isn't a visual-ambiguity exclusion like 0/O/I/l — it's a hard mathematical \
             constraint from how the version byte encodes in Base58.",
            prefix,
            chars[1],
            VALID_SECOND_CHARS.len(),
            VALID_SECOND_CHARS
        ));
    }
    Err(format!(
        "'{}' can NEVER be found.\n  This specific combination of characters is mathematically \
         unreachable, even though every individual character looks valid on its own — VerusCoin's \
         fixed version byte constrains which multi-character sequences are possible in ways that go \
         beyond any single character in isolation. Try a different combination.",
        prefix
    ))
}

/// Validates every prefix: the general Base58 alphabet rules (shared
/// with suffix/infix validation) plus the achievability check above.
/// Exits with every problem found, same as `validate_all_or_exit`.
fn validate_all_prefixes_or_exit(prefixes: &[String]) {
    let mut had_error = false;
    for prefix in prefixes {
        if let Err(msg) = validate_pattern(prefix) {
            eprintln!("⚠️  Invalid prefix: {}", msg);
            had_error = true;
            continue; // the achievability check would likely just pile on
        }
        if let Err(msg) = validate_prefix_achievable(prefix) {
            eprintln!("⚠️  Invalid prefix: {}", msg);
            had_error = true;
        }
    }
    if had_error {
        eprintln!("\nAborting — fix the prefix(es) above and try again.");
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

/// True if an argument is meant as a path rather than a literal pattern.
/// A valid Base58 pattern can never contain '.', '/' or '\', so those
/// characters unambiguously indicate a filename. Shared by
/// `read_patterns` and the startup banner so the two can never disagree
/// about how an argument was interpreted.
fn looks_like_path(arg: &str) -> bool {
    arg.contains('.') || arg.contains('/') || arg.contains('\\')
}

/// Reads a CLI value that is either a literal pattern or a path to a
/// file of patterns (one per line).
///
/// The two are told apart by content, not by whether a file happens to
/// exist — see `looks_like_path`. Deciding purely on "does a file with
/// this name exist" would mean `-p RCA` silently reads a file instead of
/// searching for the prefix, just because a file named RCA happened to
/// be sitting in the working directory.
fn read_patterns(arg: &str) -> Vec<String> {
    if !looks_like_path(arg) {
        return vec![arg.to_string()];
    }

    if !std::path::Path::new(arg).exists() {
        eprintln!(
            "⚠️  '{}' looks like a filename (it contains '.', '/' or '\\') but no such file exists.\n  \
             Base58 patterns can never contain those characters, so this was not treated as a \
             literal pattern. Check the path, or drop those characters if you meant a pattern.",
            arg
        );
        std::process::exit(1);
    }

    match std::fs::read_to_string(arg) {
        Ok(contents) => contents
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect(),
        Err(e) => {
            eprintln!("⚠️  Could not read pattern file '{}': {}", arg, e);
            std::process::exit(1);
        }
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

/// Same as `match_any_prefix` but for infixes — matches anywhere in the
/// address (`contains`), not just at the start or end.
fn match_any_infix<'a>(infixes: &'a [String], addr: &str) -> Option<Option<&'a str>> {
    if infixes.is_empty() {
        return Some(None);
    }
    infixes
        .iter()
        .find(|i| addr.contains(i.as_str()))
        .map(|i| Some(i.as_str()))
}

/// Builds the human-readable "for prefix 'X' + infix 'Y' + suffix 'Z'"
/// description used in match reports, covering every combination of
/// which constraint(s) were actually in play.
fn describe_match(prefix_hit: Option<&str>, infix_hit: Option<&str>, suffix_hit: Option<&str>) -> String {
    let mut parts = Vec::new();
    if let Some(p) = prefix_hit {
        parts.push(format!("prefix '{}'", p));
    }
    if let Some(i) = infix_hit {
        parts.push(format!("infix '{}'", i));
    }
    if let Some(s) = suffix_hit {
        parts.push(format!("suffix '{}'", s));
    }
    if parts.is_empty() {
        "match".to_string()
    } else {
        parts.join(" + ")
    }
}

// ===================== Probability & time estimates =====================

/// Estimates the expected number of addresses that must be generated
/// before at least one (prefix, suffix, infix) combination matches, on
/// average. Prefixes count only characters beyond the guaranteed leading
/// 'R'; suffixes have no guaranteed characters, so all of them count;
/// infixes can match starting at any of (34 - length + 1) positions, so
/// their probability is scaled up accordingly (each position treated as
/// roughly independent — the same approximation already used elsewhere
/// here). Combined probability across categories is approximated as the
/// product of their individual probabilities (treating leading,
/// trailing, and arbitrary-position windows as independent).
fn expected_tries(prefixes: &[String], suffixes: &[String], infixes: &[String]) -> f64 {
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
    let infix_probs: Vec<f64> = if infixes.is_empty() {
        vec![1.0]
    } else {
        infixes
            .iter()
            .map(|i| {
                let len = i.len();
                if len == 0 || len > ADDR_CHARS {
                    return 0.0;
                }
                let positions = (ADDR_CHARS - len + 1) as f64;
                (positions * BASE58_ALPHABET_SIZE.powi(-(len as i32))).min(1.0)
            })
            .collect()
    };

    let mut total_probability = 0.0f64;
    for pp in &prefix_probs {
        for sp in &suffix_probs {
            for ip in &infix_probs {
                total_probability += pp * sp * ip;
            }
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

// ===================== Prefix & suffix pre-filters (speed) =====================
//
// Base58 encodes the entire 25-byte address as one big number, so the
// LEADING characters are dominated by the high-order bytes (version +
// start of hash160), while the checksum — the last 4 bytes, unknowable
// without actually hashing — only ever perturbs the number by at most
// 2^32 out of a ~2^200 range. That gap is what makes prefix pre-filtering
// safe: for any candidate, [first21 as a number with checksum=0, first21
// with checksum=0xFFFFFFFF] is a small, fully-known range. If that whole
// range falls outside the numeric window that would produce the desired
// prefix, NO possible checksum could ever make it match — proven by
// exact integer bounds, not approximation, so it can be skipped before
// ever computing the checksum or encoding to Base58.
//
// Suffix matching seems at first like it can't use any shortcut — the
// checksum IS the primary thing determining the trailing characters, not
// a small perturbation. That's actually still true for SHORT suffixes:
// for a suffix of length k, the checksum's ~4.3 billion possible values
// cover the *entire* space of 58^k possible endings whenever 58^k <=
// 2^32 (true for k <= 5) — meaning literally any suffix is reachable
// from any candidate, so no filtering is mathematically possible there.
// But for k >= 6, 58^k exceeds 2^32, so the checksum's range no longer
// covers every possibility — a real, provable filter becomes available
// using modular arithmetic on (first21 * 2^32) mod 58^k. See
// `provably_outside_suffix` for the exact reasoning. Validated against
// 27 million random samples across suffix lengths 6-10 with zero false
// negatives before shipping, with skip rates matching the predicted
// 1 - 2^32/58^k almost exactly (e.g. ~88.7% for k=6).
//
// Neither filter can ever cause a wrong result: the final match decision
// always goes through the same exact, unchanged `match_any_prefix` /
// `match_any_suffix` + Base58 comparison as before. A filter only ever
// skips candidates already *proven* impossible for that one category —
// and since a match requires every present category (prefix AND suffix
// AND infix) to be satisfied, proving just one of them impossible is
// already enough to safely skip, regardless of what the others would
// say. If bound computation fails or is unavailable (infix always lacks
// one; suffixes under 6 characters mathematically can't have one), that
// category simply never contributes to the skip decision, and candidates
// fall through to the always-correct exact path for it — same as before
// either optimization existed.

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
/// empty Vec (meaning "no prefix-based skip available") if any single
/// prefix's bound can't be computed, or if there are no prefixes at all.
/// All-or-nothing across the prefix list keeps the per-candidate check
/// simple: either every prefix has a valid bound, or none of them get
/// the fast-path treatment. Independent of whether suffixes or infixes
/// are also present — see the module note above for why that's safe.
fn try_compute_all_prefix_bounds(prefixes: &[String]) -> Vec<PrefixBound> {
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
fn provably_outside_prefix(first21: &[u8; 21], bound: &PrefixBound) -> bool {
    let mut lo = [0u8; 25];
    lo[0..21].copy_from_slice(first21);
    let mut hi = [0u8; 25];
    hi[0..21].copy_from_slice(first21);
    hi[21..25].copy_from_slice(&[0xff, 0xff, 0xff, 0xff]);

    hi < bound.lower || lo >= bound.upper_exclusive
}

// ----- Suffix pre-filter (length >= 6 only — see module note above) -----

/// Numeric bound for fast suffix pre-filtering. `modulus` is 58^k for a
/// suffix of length k; `target` is the suffix parsed as a base-58
/// integer (each character a standard digit, positionally weighted).
#[derive(Clone, Copy)]
struct SuffixBound {
    modulus: u64,
    target: u64,
}

/// Parses a pattern as a base-58 integer — NOT the same as
/// `bs58::decode`, which has extra byte-array semantics (like leading-'1'
/// handling) we specifically don't want here. Each character is just a
/// standard positional digit 0-57.
fn parse_base58_number(s: &str) -> Option<u64> {
    let mut value: u64 = 0;
    for c in s.bytes() {
        let digit = BASE58_ALPHABET.bytes().position(|x| x == c)? as u64;
        value = value.checked_mul(58)?.checked_add(digit)?;
    }
    Some(value)
}

/// (first21 interpreted as a big-endian integer) mod `modulus`, computed
/// byte-by-byte so it never needs a number bigger than fits in u128 —
/// avoids needing a general bignum library for this.
fn first21_mod(first21: &[u8; 21], modulus: u64) -> u64 {
    let mut r: u128 = 0;
    for &byte in first21.iter() {
        r = (r * 256 + byte as u128) % modulus as u128;
    }
    r as u64
}

/// Computes a suffix pre-filter bound. Only available for lengths 6-10
/// (see module note above for the mathematical reason 6 is the cutoff);
/// 10 is a conservative ceiling comfortably within u64 range (58^10 is
/// still ~40x smaller than u64::MAX). Returns None outside that range,
/// or on any unexpected condition — callers must treat None as "no
/// suffix-based skip available", not as an error.
fn compute_suffix_bound(suffix: &str) -> Option<SuffixBound> {
    let k = suffix.len();
    if !(6..=10).contains(&k) {
        return None;
    }
    let modulus = 58u64.checked_pow(k as u32)?;
    let target = parse_base58_number(suffix)?;
    if target >= modulus {
        return None;
    }
    Some(SuffixBound { modulus, target })
}

/// Attempts to compute a pre-filter bound for every suffix. Empty Vec
/// means "no suffix-based skip available" — either because a suffix is
/// shorter than 6 characters (mathematically impossible to filter, not
/// just unimplemented) or because there are no suffixes at all.
fn try_compute_all_suffix_bounds(suffixes: &[String]) -> Vec<SuffixBound> {
    let mut bounds = Vec::with_capacity(suffixes.len());
    for s in suffixes {
        match compute_suffix_bound(s) {
            Some(b) => bounds.push(b),
            None => return Vec::new(),
        }
    }
    bounds
}

/// Returns true only if literally no checksum value (0..2^32) could make
/// this candidate's address end with the suffix behind `bound`. See the
/// module note above for the full reasoning; in short: the checksum's
/// range no longer covers every possible ending once the suffix is 6+
/// characters, so for a given candidate only a specific reachable window
/// of endings is possible, computed here via modular arithmetic.
fn provably_outside_suffix(first21: &[u8; 21], bound: &SuffixBound) -> bool {
    // (first21_num * 2^32) mod modulus. Since modulus > 2^32 always in
    // our supported range (k >= 6), 2^32 mod modulus is just 2^32
    // itself, so no extra reduction of the multiplier is needed.
    let a = first21_mod(first21, bound.modulus);
    let base_remainder = ((a as u128 * (1u128 << 32)) % bound.modulus as u128) as u64;

    // How far around the modulus circle from base_remainder to target.
    // If some checksum c in [0, 2^32) reaches target, c must equal this
    // distance exactly — modulus exceeds 2^32 throughout our supported
    // range, so there's at most one representative of this residue class
    // small enough to be a valid checksum value.
    let diff = bound.target as i128 - base_remainder as i128;
    let forward_distance = diff.rem_euclid(bound.modulus as i128) as u64;

    forward_distance >= (1u64 << 32)
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
    infixes: Vec<String>,
    prefix_arg: Option<String>,
    suffix_arg: Option<String>,
    infix_arg: Option<String>,
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
        .version("0.6.2")
        .author("Darktron")
        .about("VerusCoin Vanity Wallet Generator\nMade by Darktron")
        .disable_version_flag(true)
        .arg(
            Arg::new("prefix")
                .short('p')
                .long("prefix")
                .help("Prefix string or filename with prefixes (one per line)")
                .num_args(1),
        )
        .arg(
            Arg::new("infix")
                .short('i')
                .long("infix")
                .help("Infix string or filename with infixes (one per line)")
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
                // Without this, clap treats the "-1" in `-m -1` as a flag
                // and errors out — even though the help text (and the
                // default) advertise exactly that value.
                .allow_hyphen_values(true)
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
        .arg(
            Arg::new("version")
                .short('v')
                .long("version")
                .action(ArgAction::Version)
                .help("Print version"),
        )
        .get_matches();

    let prefix_arg = matches.get_one::<String>("prefix").cloned();
    let suffix_arg = matches.get_one::<String>("suffix").cloned();
    let infix_arg = matches.get_one::<String>("infix").cloned();

    if prefix_arg.is_none() && suffix_arg.is_none() && infix_arg.is_none() {
        eprintln!("⚠️  Provide at least one of --prefix/-p, --suffix/-s, or --infix/-i.");
        std::process::exit(1);
    }

    let output_file = matches.get_one::<String>("output").cloned();
    // Clamp to at least 1: `-t 0` would otherwise spawn no workers and
    // exit immediately without searching or explaining why.
    let threads: usize = matches
        .get_one::<String>("threads")
        .unwrap()
        .parse()
        .unwrap_or(num_cpus::get())
        .max(1);
    let max_matches: i64 = matches
        .get_one::<String>("matches")
        .unwrap()
        .parse()
        .unwrap_or(-1);
    // -1 means infinite; anything else must be a positive count. Without
    // this, `-m 0` (or any value below -1) satisfied the stop condition
    // instantly and the program exited having done nothing.
    if max_matches == 0 || max_matches < -1 {
        eprintln!(
            "⚠️  --matches/-m must be a positive number, or -1 for infinite (got {}).",
            max_matches
        );
        std::process::exit(1);
    }

    let raw_prefixes: Vec<String> = prefix_arg.as_deref().map(read_patterns).unwrap_or_default();
    let suffixes: Vec<String> = suffix_arg.as_deref().map(read_patterns).unwrap_or_default();
    let infixes: Vec<String> = infix_arg.as_deref().map(read_patterns).unwrap_or_default();

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

    validate_all_prefixes_or_exit(&prefixes);
    validate_all_or_exit(&suffixes, "suffix");
    validate_all_or_exit(&infixes, "infix");

    Config {
        raw_prefixes,
        prefixes,
        suffixes,
        infixes,
        prefix_arg,
        suffix_arg,
        infix_arg,
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
            if looks_like_path(a) {
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
        println!("Prefixes: none");
    }

    if !config.infixes.is_empty() {
        if let Some(a) = &config.infix_arg {
            if looks_like_path(a) {
                println!(
                    "Infix file: {}",
                    std::fs::canonicalize(a).unwrap_or_else(|_| a.into()).display()
                );
            }
        }
        println!("Infixes:");
        for i in &config.infixes {
            println!("  {}", i);
        }
    } else {
        println!("Infixes: none");
    }

    if !config.suffixes.is_empty() {
        if let Some(a) = &config.suffix_arg {
            if looks_like_path(a) {
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
        println!("Suffixes: none");
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
    /// Serializes match reporting. Each report is many lines (address,
    /// WIF, key, and a whole QR code); without this, two threads finding
    /// matches at the same moment interleave their lines and produce an
    /// unreadable, unscannable mess on the terminal.
    report_lock: Arc<Mutex<()>>,
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
            report_lock: Arc::new(Mutex::new(())),
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
fn report_match(found_num: i64, desc: &str, addr: &str, wif: &str, priv_hex: &str, state: &SharedState) {
    let qr = wif_to_qr_string(wif);
    let body = format!(
        "----- MATCH {} for {} FOUND -----\nAddress: {}\nWIF: {}\nPrivate Key (hex): {}\n\
         Scan this QR code to import the WIF into your wallet app:\n{}\n-------------------------\n",
        found_num, desc, addr, wif, priv_hex, qr
    );

    // Held across both the terminal write and the file write so a
    // concurrent match can't interleave with either.
    let _guard = state.report_lock.lock().unwrap_or_else(|e| e.into_inner());

    print!("{}\n", body);
    let _ = std::io::stdout().flush();

    if let Some(output_mutex) = &state.output_writer {
        let mut output = output_mutex.lock().unwrap_or_else(|e| e.into_inner());
        write!(output, "{}\n", body).ok();
        output.flush().ok();
    }
}

// ===================== Optional performance-core targeting =====================
//
// Many phone SoCs mix fast "performance" cores with slower "efficiency"
// cores (big.LITTLE / DynamIQ). Left to the OS scheduler, some worker
// threads can end up running on the slow cores, diluting average
// throughput — especially when -t is set below the total core count.
// This tries to identify the fastest cores (via each core's max clock
// speed, read from sysfs) and pin worker threads to them specifically,
// fastest-first.
//
// This is entirely best-effort and safe to fail: if core IDs can't be
// enumerated, if ANY core's frequency can't be read, or if pinning
// itself fails, this quietly does nothing and threads run exactly as
// they did before this existed — plain OS scheduling, no behavior
// change, no risk. It never touches anything cryptographic.

/// Reads a core's maximum clock speed from sysfs (Linux/Android/Termux).
/// Returns None if unavailable — e.g. in containers/VMs, or on any
/// platform that doesn't expose this, which is common and expected.
fn core_max_freq_khz(core_id: usize) -> Option<u64> {
    let path = format!("/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_max_freq", core_id);
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Attempts to order available cores fastest-first. Returns None (rather
/// than a partially-correct guess) unless every core's frequency was
/// readable — a partial ordering could be misleading, so this only ever
/// acts on complete information.
fn fastest_cores_first() -> Option<Vec<core_affinity::CoreId>> {
    let core_ids = core_affinity::get_core_ids()?;
    let mut with_freq = Vec::with_capacity(core_ids.len());
    for core_id in core_ids {
        let freq = core_max_freq_khz(core_id.id)?;
        with_freq.push((core_id, freq));
    }
    with_freq.sort_by_key(|&(_, freq)| std::cmp::Reverse(freq));
    Some(with_freq.into_iter().map(|(core_id, _)| core_id).collect())
}


// ===================== Batched affine point generation =====================
//
// The previous approach walked a sequential chain in projective
// coordinates (each point derived from the one before), then converted
// the whole batch to affine with one shared inversion. That works, but
// every step still paid a full projective addition (~11 field
// multiplications) plus its share of the normalization.
//
// This approach restructures the work so the additions are INDEPENDENT
// rather than sequential: precompute a fixed table of G, 2G, 3G, ..., NG
// once at startup, then from a single base point P compute P+G, P+2G,
// ..., P+NG. Because none of those depend on each other, they can all be
// done directly in affine coordinates, and every one of their required
// inversions collapses into a single batched inversion (Montgomery's
// trick). That drops the per-point cost to roughly 6 field
// multiplications instead of ~15.
//
// Measured 2.25x faster on the EC math in isolation, and validated
// bit-identical against k256's own projective arithmetic across
// thousands of points — including under a debug build, where k256's
// internal field-element magnitude assertions are active and would catch
// any error in the normalization discipline below.

/// Extracts (x, y) field elements from an affine point by round-tripping
/// through its uncompressed SEC1 encoding. Only used at startup and once
/// per batch for the base point — never per candidate.
fn affine_xy(p: &AffinePoint) -> Option<(FieldElement, FieldElement)> {
    let enc = p.to_encoded_point(false);
    let bytes = enc.as_bytes();
    if bytes.len() != 65 {
        return None; // identity point or unexpected encoding
    }
    let x = FieldElement::from_bytes(bytes[1..33].into());
    let y = FieldElement::from_bytes(bytes[33..65].into());
    if bool::from(x.is_some()) && bool::from(y.is_some()) {
        Some((x.unwrap(), y.unwrap()))
    } else {
        None
    }
}

/// Builds the compressed SEC1 public key (0x02/0x03 || x) from raw
/// coordinates — this is exactly what gets hashed to form the address.
fn compressed_from_xy(x: &FieldElement, y: &FieldElement) -> [u8; 33] {
    let xn = x.normalize();
    let yn = y.normalize();
    let mut out = [0u8; 33];
    out[0] = if bool::from(yn.is_odd()) { 0x03 } else { 0x02 };
    out[1..33].copy_from_slice(&xn.to_bytes());
    out
}

/// Precomputes the addition table: entry i holds the affine coordinates
/// of (i+1)*G. Built once at startup and shared (by clone) with every
/// worker thread.
fn build_addition_table() -> Vec<(FieldElement, FieldElement)> {
    let mut table = Vec::with_capacity(BATCH_SIZE);
    let mut acc = ProjectivePoint::GENERATOR;
    for _ in 0..BATCH_SIZE {
        let xy = affine_xy(&acc.to_affine()).expect("multiples of G are always valid affine points");
        table.push(xy);
        acc += ProjectivePoint::GENERATOR;
    }
    table
}

/// Allocation-free Montgomery batch inversion. Inverts `values` in
/// place, using `scratch` (same length) for prefix products. Both
/// buffers are owned by the caller and reused across every batch, so
/// this performs zero heap allocation per call — unlike the library's
/// `batch_invert`, which allocates and zeroes three buffers of this size
/// on every single call and then copies the result out again.
///
/// Returns false if any element is zero (leaving `values` unspecified),
/// exactly like the library version returning None. Callers must treat
/// that as "cannot proceed", never as success.
///
/// Validated against the library implementation across 102,400 random
/// elements with zero mismatches, with every result additionally checked
/// to satisfy a * a⁻¹ == 1, and confirmed to reject a zero input.
fn batch_invert_in_place(values: &mut [FieldElement], scratch: &mut [FieldElement]) -> bool {
    debug_assert_eq!(values.len(), scratch.len());
    let n = values.len();
    if n == 0 {
        return true;
    }

    // Forward pass: scratch[i] = product of values[0..i]
    let mut acc = FieldElement::ONE;
    for i in 0..n {
        scratch[i] = acc;
        acc = acc * values[i];
    }

    // One inversion for the entire batch — the whole point of the trick.
    let total_inv = acc.invert();
    if !bool::from(total_inv.is_some()) {
        return false;
    }
    let mut inv = total_inv.unwrap();

    // Backward pass, writing each result in place. `original` must be
    // captured before the overwrite, since it's still needed to advance
    // the running inverse.
    for i in (0..n).rev() {
        let original = values[i];
        values[i] = inv * scratch[i];
        inv = inv * original;
    }
    true
}

/// One worker thread's search loop — see the module note above for the
/// batched-affine generation strategy.
fn worker_loop(prefixes: Vec<String>, suffixes: Vec<String>, infixes: Vec<String>, prefix_bounds: Vec<PrefixBound>, suffix_bounds: Vec<SuffixBound>, max_matches: i64, state: SharedState, pinned_core: Option<core_affinity::CoreId>, table: Vec<(FieldElement, FieldElement)>) {
    // Best-effort only — see the module note above `fastest_cores_first`.
    // If this is None, or if pinning fails on this particular platform,
    // the thread just runs under normal OS scheduling, exactly as before
    // this feature existed.
    if let Some(core_id) = pinned_core {
        core_affinity::set_for_current(core_id);
    }

    let mut rng_source = thread_rng();
    let mut rng = ChaCha20Rng::from_rng(&mut rng_source).expect("failed to seed RNG");

    // Advancing the base by BATCH_SIZE each round keeps `base_scalar` and
    // `base_proj` in lockstep: base_proj is always base_scalar * G. Every
    // reported match re-derives its address from the scalar and checks it
    // against the matched address before printing, so any drift between
    // the two would be caught rather than producing a bad key.
    let step = ProjectivePoint::GENERATOR * Scalar::from(BATCH_SIZE as u64);
    let mut base_scalar = Scalar::random(&mut rng);
    let mut base_proj = ProjectivePoint::GENERATOR * base_scalar;
    let mut offset: u64 = 0;

    // Both reused across every batch — no per-batch allocation.
    let mut denominators = [FieldElement::ONE; BATCH_SIZE];
    let mut invert_scratch = [FieldElement::ONE; BATCH_SIZE];

    'outer: loop {
        if max_matches != -1 && state.found_count.load(Ordering::Relaxed) >= max_matches {
            break;
        }

        let (px, py) = match affine_xy(&base_proj.to_affine()) {
            Some(xy) => xy,
            None => {
                // Base landed on the identity — astronomically unlikely.
                // Just pick a fresh random base and carry on.
                base_scalar = Scalar::random(&mut rng);
                base_proj = ProjectivePoint::GENERATOR * base_scalar;
                offset = 0;
                continue;
            }
        };

        // Subtraction leaves a small magnitude that's already well within
        // what mul accepts, so no normalize_weak is needed here — the
        // debug build's magnitude assertions verify this.
        for (slot, entry) in denominators.iter_mut().zip(table.iter()) {
            *slot = entry.0 - px;
        }

        // Inverts in place, reusing both buffers — no allocation. Fails
        // only if some table point shares an x-coordinate with the base
        // (probability ~2^-256); rebase rather than risk anything.
        if !batch_invert_in_place(&mut denominators, &mut invert_scratch) {
            base_scalar = Scalar::random(&mut rng);
            base_proj = ProjectivePoint::GENERATOR * base_scalar;
            offset = 0;
            continue;
        }
        let inverses = &denominators;

        for i in 0..BATCH_SIZE {
            if max_matches != -1 && state.found_count.load(Ordering::Relaxed) >= max_matches {
                break 'outer;
            }

            let (tx, ty) = table[i];
            // Standard affine addition: λ = (y2-y1)/(x2-x1),
            // x3 = λ² - x1 - x2, y3 = λ(x1-x3) - y1.
            //
            // No normalize_weak on lambda or y_new: mul and square both
            // return magnitude-1 results, and the intermediate
            // subtractions stay within the budget mul accepts. x_new DOES
            // need one, because subtraction in k256 requires its
            // right-hand side to be magnitude-1 and x_new is used that
            // way just below. All of this is verified by running the
            // debug build, where k256's internal magnitude assertions are
            // active — they caught exactly this case when it was missing.
            let lambda = (ty - py) * inverses[i];
            let x_new = (lambda.square() - px - tx).normalize_weak();
            let y_new = lambda * (px - x_new) - py;

            // Index within this batch: table[i] is (i+1)*G, and
            // base_scalar already advances by BATCH_SIZE each round, so
            // the key for this candidate is base_scalar + (i+1). (The
            // separate `offset` counter below only tracks progress
            // toward the periodic rebase — adding it here too would
            // double-count.)
            let this_offset = (i + 1) as u64;
            let compressed = compressed_from_xy(&x_new, &y_new);
            let first21 = compute_first21(&compressed, ADDRESS_VERSION_BYTE);

            // Fast path: a match requires every present category
            // (prefix AND suffix AND infix) to be satisfied, so proving
            // just ONE of them impossible is already enough to skip the
            // checksum computation and Base58 encode entirely —
            // regardless of what the others would say. Each bound set
            // is empty (and so never triggers) whenever that category
            // has no patterns, or lacks a usable bound (infix always
            // lacks one; suffixes under 6 characters mathematically
            // can't have one) — candidates then fall through to the
            // exact path below for that category, same as before either
            // optimization existed.
            let prefix_proves_impossible =
                !prefix_bounds.is_empty() && prefix_bounds.iter().all(|b| provably_outside_prefix(&first21, b));
            let suffix_proves_impossible =
                !suffix_bounds.is_empty() && suffix_bounds.iter().all(|b| provably_outside_suffix(&first21, b));
            if prefix_proves_impossible || suffix_proves_impossible {
                continue;
            }

            let addr = address_from_first21(&first21);

            let prefix_hit = match_any_prefix(&prefixes, &addr);
            let suffix_hit = match_any_suffix(&suffixes, &addr);
            let infix_hit = match_any_infix(&infixes, &addr);

            if let (Some(prefix_hit), Some(suffix_hit), Some(infix_hit)) = (prefix_hit, suffix_hit, infix_hit) {
                // Only reconstruct the actual private key on a real match
                // (rare) — the hot loop above never needs it, since it
                // only ever walks the public key forward.
                let sk_scalar = base_scalar + Scalar::from(this_offset);
                let sk_bytes: [u8; 32] = sk_scalar.to_bytes().into();

                // SELF-CHECK (defense in depth): independently re-derive
                // the address straight from the reconstructed private key
                // using k256's own scalar multiplication — a completely
                // different code path from the batched affine arithmetic
                // that produced this candidate. If they disagree, the key
                // would not control the address, so refuse to report it
                // rather than hand out something unusable. This costs one
                // scalar multiplication, but only ever on a real match.
                let check_point = ProjectivePoint::GENERATOR * sk_scalar;
                let check_compressed = check_point.to_affine().to_encoded_point(true);
                let check_addr = address_from_first21(&compute_first21(
                    check_compressed.as_bytes(),
                    ADDRESS_VERSION_BYTE,
                ));
                if check_addr != addr {
                    eprintln!(
                        "⚠️  Internal consistency check FAILED for a candidate address — \
                         discarding it rather than reporting a key that would not control it. \
                         Please report this; the search continues."
                    );
                    continue;
                }

                let wif = private_key_to_wif(&sk_bytes, WIF_VERSION_BYTE, true);
                let priv_hex = hex::encode(sk_bytes);
                let desc = describe_match(prefix_hit, infix_hit, suffix_hit);

                // Atomically claim a slot. Checking the count and then
                // incrementing it as two separate steps let two threads
                // both pass the check and both report, overshooting
                // -m N. Claiming first and validating the claimed slot
                // number makes that impossible: only claims below the
                // limit are reported, and an over-limit claim is handed
                // straight back.
                let claimed = state.found_count.fetch_add(1, Ordering::Relaxed);
                if max_matches != -1 && claimed >= max_matches {
                    state.found_count.fetch_sub(1, Ordering::Relaxed);
                    break 'outer;
                }
                let found_num = claimed + 1;
                // Reset the progress-percentage baseline to start counting
                // fresh from this match forward. fetch_max (rather than a
                // plain store) keeps this correct even if two threads find
                // matches close together and race here.
                state
                    .last_match_tries
                    .fetch_max(state.keys_tried.load(Ordering::Relaxed), Ordering::Relaxed);

                report_match(found_num, &desc, &addr, &wif, &priv_hex, &state);
            }
        }

        // Advance both representations of the base together, so
        // base_proj stays equal to base_scalar * G.
        base_scalar += Scalar::from(BATCH_SIZE as u64);
        base_proj += step;
        offset += BATCH_SIZE as u64;
        state.keys_tried.fetch_add(BATCH_SIZE as u64, Ordering::Relaxed);

        if offset >= REBASE_INTERVAL {
            base_scalar = Scalar::random(&mut rng);
            base_proj = ProjectivePoint::GENERATOR * base_scalar;
            offset = 0;
        }
    }
}

// ===================== Entry point =====================

fn main() {
    let config = parse_cli();
    let expected = expected_tries(&config.prefixes, &config.suffixes, &config.infixes);
    print_banner(&config);

    // Computed once each, independently, reused read-only by every
    // thread. Either can be empty (meaning "no skip available from this
    // category") without affecting the other — see the module note on
    // the pre-filter section for why that's safe.
    let prefix_bounds = try_compute_all_prefix_bounds(&config.prefixes);
    let suffix_bounds = try_compute_all_suffix_bounds(&config.suffixes);

    // Best-effort: None if unavailable on this platform (common — most
    // desktop/server systems and many phones don't expose this), in
    // which case every thread below just gets None and runs under
    // normal OS scheduling.
    let core_order = fastest_cores_first();

    // Precompute G, 2G, ..., BATCH_SIZE*G once; each thread gets its own
    // copy (~32KB) so the hot loop reads it without any shared-pointer
    // indirection.
    let table = build_addition_table();

    let state = SharedState::new(&config.output_file);
    let start_time = Instant::now();
    let target_matches: f64 = if config.max_matches == -1 { 1.0 } else { config.max_matches.max(1) as f64 };

    spawn_stats_thread(state.clone(), start_time, expected, target_matches);

    let mut handles = Vec::new();
    for i in 0..config.threads {
        let prefixes = config.prefixes.clone();
        let suffixes = config.suffixes.clone();
        let infixes = config.infixes.clone();
        let prefix_bounds = prefix_bounds.clone();
        let suffix_bounds = suffix_bounds.clone();
        let state = state.clone();
        let max_matches = config.max_matches;
        // Fastest core first, cycling if there are more threads than
        // detected cores (e.g. -t set above the core count).
        let pinned_core = core_order.as_ref().map(|order| order[i % order.len()]);
        let table = table.clone();
        handles.push(thread::spawn(move || {
            worker_loop(prefixes, suffixes, infixes, prefix_bounds, suffix_bounds, max_matches, state, pinned_core, table)
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
