//! verus-vanity — a VerusCoin vanity wallet address generator.
//!
//! Searches for a private key whose derived transparent ("R…") address
//! matches a desired prefix, infix, and/or suffix.
//!
//! Candidates are produced in batches of independent affine point
//! additions from a precomputed table, sharing one modular inversion per
//! batch (Montgomery's trick) — see "Batched affine point generation"
//! below. Each computed point then yields six addresses for almost no
//! extra work via curve symmetry (see "GLV endomorphism & negation
//! symmetry"). Their RIPEMD-160 hashes are computed several at a time
//! through a multi-buffer implementation (see "Multi-buffer RIPEMD-160").
//! Most candidates are then discarded by numeric pre-filters
//! before their checksum is ever computed (see "Prefix & suffix
//! pre-filters"). Every reported match is re-derived from its private
//! key through an independent code path and checked before being
//! printed, so a key that would not control its address is never
//! emitted.
//!
//! `--bench` reports what each stage of that pipeline costs on the
//! machine it is run on. Several constants here are tuned to measured
//! numbers rather than to theory — more than one plausible-sounding
//! prediction turned out backwards — so re-measure before changing any of
//! them, and record what came back.

use clap::{Arg, ArgAction, Command};
use ff::{Field, PrimeField};
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::{AffinePoint, FieldElement, ProjectivePoint, Scalar};
use rand::thread_rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use ripemd::Ripemd160;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
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

/// How many candidate keys are produced per batch.
///
/// Batching exists to amortize the one modular inversion each batch
/// needs across many points, so throughput rises with size until the
/// working set stops fitting in cache. MEASURED on a Snapdragon 8 Elite
/// (8 threads, median of 5, addresses/second):
///
///      64 -> 37.2      2048 -> 45.6     32768 -> 45.5
///     128 -> 41.1      4096 -> 45.8     65536 -> 47.0
///     256 -> 43.5      8192 -> 45.8    131072 -> 45.6
///     512 -> 44.6     16384 -> 45.5    262144 -> 43.1
///
/// It rises steeply to about 2048, is flat from there to roughly 8192,
/// and falls away past 131072. A real search confirmed the difference at
/// the ends: 43.6 MW/s sustained at 480, 44.8 at 4096 — +2.8%, with mean
/// and median agreeing.
///
/// So the default sits in that plateau. Why 3840 rather than 4096: the
/// assertions below require BATCH_SIZE * 2 and * 6 to divide evenly by
/// the lane count, because candidates are staged in lane groups and a
/// partial group at the end of a batch would be silently skipped. 4096 is
/// 2^12, so it is not divisible by 12 — a `VERUS_VECS=3` build would fail
/// to compile, and `tune-lanes.sh` builds exactly that. 3840 is
/// 2^8 x 3 x 5, which covers every lane width either backend can be
/// built with, and sits inside the flat region where 4096 measured
/// identically anyway.
///
/// The previous default of 480 was chosen only for that divisibility, not
/// for throughput — it sits well down the rising part of the curve.
///
/// Overridable per run with `-b`, which is worth doing on a device with a
/// small shared cache: a big batch there can thrash it.
const BATCH_SIZE: usize = 3840;

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
/// Interprets a 25-byte big-endian value as f64. Only used for
/// probability estimates, where f64's ~15 significant digits are far more
/// precision than an ETA needs.
fn bytes25_to_f64(b: &[u8; 25]) -> f64 {
    let mut v = 0.0f64;
    for &x in b.iter() {
        v = v * 256.0 + x as f64;
    }
    v
}

/// Exact probability that a random VerusCoin address starts with
/// `prefix`.
///
/// The obvious formula — 58^-(len-1), one free character for the
/// guaranteed leading 'R' — is wrong, and not by a little. Address values
/// only ever occupy [version * 2^192, (version+1) * 2^192), which covers
/// just 24 of the 58 possible second-character buckets, so that character
/// is about 2.4x more likely to match than 1/58 suggests. For `-p RCAB`
/// the old formula predicted 195,112 tries against a true expectation of
/// 78,509 — a 2.49x overestimate, which showed up as searches finishing
/// far sooner than the ETA claimed.
///
/// This instead measures how much of the achievable value range the
/// prefix's own Base58 bucket covers, reusing the exact bounds already
/// computed for the pre-filter. Returns None if the bucket cannot be
/// computed, leaving the caller to fall back to the rough formula.
fn prefix_probability(prefix: &str) -> Option<f64> {
    let bound = compute_prefix_bound(prefix)?;
    let bucket_lo = bytes25_to_f64(&bound.lower);
    let bucket_hi = bytes25_to_f64(&bound.upper_exclusive);

    let span = 2f64.powi(192);
    let achievable_lo = ADDRESS_VERSION_BYTE as f64 * span;
    let achievable_hi = (ADDRESS_VERSION_BYTE as f64 + 1.0) * span;

    let overlap = (bucket_hi.min(achievable_hi) - bucket_lo.max(achievable_lo)).max(0.0);
    Some(overlap / (achievable_hi - achievable_lo))
}

fn expected_tries(prefixes: &[String], suffixes: &[String], infixes: &[String]) -> f64 {
    let prefix_probs: Vec<f64> = if prefixes.is_empty() {
        vec![1.0]
    } else {
        prefixes
            .iter()
            .map(|p| {
                // Exact where possible; the rough 58^-(len-1) form only as
                // a fallback. See `prefix_probability`.
                prefix_probability(p)
                    .unwrap_or_else(|| BASE58_ALPHABET_SIZE.powi(-(p.len().saturating_sub(1) as i32)))
            })
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

// ===================== Multi-buffer RIPEMD-160 =====================
//
// With SHA-256 running on the ARMv8 crypto extension, RIPEMD-160 — which
// has no hardware acceleration on any target — became roughly two thirds
// of the per-address cost. It cannot be made faster for a single hash,
// but it parallelizes perfectly ACROSS hashes: the whole algorithm is
// 32-bit add / rotate / and / or / xor with no data-dependent branching
// and no lookups, so N independent hashes run as N lanes of the same
// instruction stream.
//
// This is written as portable safe Rust over `[u32; RIPEMD_LANES]`
// rather than as NEON intrinsics: LLVM maps fixed-size u32 array
// arithmetic straight onto vector registers, so `-C target-cpu=native`
// produces the same NEON code with no `unsafe` and no aarch64-specific
// paths to maintain. If a target cannot vectorize it, it degrades to
// plain scalar work that is still correct.
//
// The constant tables below were validated before being written here, by
// implementing RIPEMD-160 from these exact tables independently and
// checking it against a separate trusted implementation over the three
// published test vectors, 200 random 32-byte inputs, and 150 random
// multi-block inputs — zero mismatches. `ripemd160_multi_matches_crate` in
// the test module re-checks this build against the `ripemd` crate.

/// Number of hashes computed side by side. Set by whichever lane backend
/// the target selected — see `lane_ops` below, where each width and the
/// reasoning behind it live.
const RIPEMD_LANES: usize = lane_ops::LANES;

/// Every batch must divide evenly into lane groups, otherwise candidates
/// would be left un-examined in the staging buffer at the end of a batch.
/// Checked at compile time for both possible variant counts (2 without
/// the endomorphism, 6 with it) so a future change to BATCH_SIZE fails to
/// build rather than silently skipping candidates.
const _: () = assert!(
    (BATCH_SIZE * 2) % RIPEMD_LANES == 0,
    "BATCH_SIZE * 2 must divide by RIPEMD_LANES (the endomorphism-off case, \
     2 addresses per point). Adjust BATCH_SIZE — it is chosen to be highly \
     composite for exactly this reason, see its doc comment."
);
const _: () = assert!(
    (BATCH_SIZE * 6) % RIPEMD_LANES == 0,
    "BATCH_SIZE * 6 must divide by RIPEMD_LANES (the endomorphism-on case, \
     6 addresses per point). Adjust BATCH_SIZE — it is chosen to be highly \
     composite for exactly this reason, see its doc comment."
);

// ----- Lane primitives, one implementation per target -----
//
// Everything above and below this point — the 160 unrolled steps, the
// round macro, the message schedule — is written once against these ten
// operations. Only the operations themselves vary by target, so adding an
// architecture means writing this short list, not touching the algorithm.
//
// MEASURED, and not what was first assumed. The portable array version
// reached 1.68x over one-hash-at-a-time hashing on aarch64; replacing it
// with the explicit NEON intrinsics below moved that to 1.62x — i.e. no
// change at all. That rules out the obvious theory: if the arrays had
// been compiling to scalar code, swapping ~6400 scalar operations for
// ~1600 vector ones would have been a large win. LLVM was already
// vectorizing them.
//
// What actually limits this is the dependency chain, not throughput. Each
// step is
//
//     t = rotl(a + f(b,c,d) + x[r] + k, s) + e
//
// which is roughly 7-8 dependent operations deep, and the next step
// cannot start until `b` receives `t`. At ~2-cycle vector ALU latency
// that is ~12 cycles per step, and 80 step-levels then cost ~990 cycles
// per group — which is exactly what was measured. Widening the vectors
// does nothing about it, because the chain is the same length however
// many lanes ride along it. The 1.6x it does achieve is simply four
// hashes sharing one chain instead of one hash owning it.
//
// So the lever is the NUMBER OF INDEPENDENT CHAINS IN FLIGHT, which is
// what `VECS` below controls: each lane group carries VECS separate
// vector registers through every step, so VECS chains interleave and fill
// each other's latency. Raising it trades register pressure for
// parallelism, and the point of diminishing returns is where the kernel
// finally becomes throughput-bound instead.
//
// Explicit intrinsics are still worth keeping despite not being the fix:
// they make the widening exact rather than something LLVM has to infer.
// Re-run `--bench` after changing anything here.

/// NEON. Mandatory in ARMv8-A, so no runtime detection is needed; the
/// `target_feature` guard only exists for the unusual builds that turn it
/// off, which fall through to the portable version below.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[allow(unused_unsafe)]
mod lane_ops {
    use core::arch::aarch64::*;

    /// Independent 128-bit vectors carried through every step, and the
    /// one number to tune here. Each one is a separate dependency chain,
    /// so this is how many chains interleave to cover each other's
    /// latency — see the analysis above for why that, and not vector
    /// width, is what this kernel is short of.
    ///
    /// MEASURED on a Snapdragon X Elite (aarch64), ns per hash:
    ///
    ///   VECS = 1  (4 lanes)   60.36     latency-bound, chains idle
    ///   VECS = 2  (8 lanes)   32.03  <- optimal
    ///   VECS = 3  (12 lanes)  43.32     spill cliff
    ///   VECS = 4  (16 lanes)  52.29     worse still
    ///
    /// Going 1 -> 2 cost 6% more time for twice the work, which is the
    /// latency slack being taken up. Going 2 -> 3 doubled the time for
    /// 1.5x the work, which is register pressure: the state alone is
    /// 5 lanes * VECS vectors * 2 lines, so VECS = 3 wants 30 of
    /// aarch64's 32 vector registers and leaves nothing for the message
    /// schedule or temporaries, and the spill traffic costs more than the
    /// extra chains buy.
    ///
    /// Do not re-tune this without re-measuring; the optimum is a narrow
    /// peak, not a plateau.
    ///
    /// Overridable at build time for devices that were never measured:
    ///     VERUS_VECS=3 cargo build --release
    const VECS: usize = super::env_usize(option_env!("VERUS_VECS"), 2);

    pub const LANES: usize = VECS * 4;

    /// VECS vectors moving in lockstep. A plain array rather than named
    /// fields so the width is changed by editing `VECS` alone; every loop
    /// over it has a constant trip count and unrolls away.
    #[derive(Clone, Copy)]
    pub struct Lane([uint32x4_t; VECS]);

    #[inline(always)]
    pub fn splat(v: u32) -> Lane {
        Lane([unsafe { vdupq_n_u32(v) }; VECS])
    }

    #[inline(always)]
    pub fn load(words: [u32; LANES]) -> Lane {
        let mut out = [unsafe { vdupq_n_u32(0) }; VECS];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = unsafe { vld1q_u32(words.as_ptr().add(i * 4)) };
        }
        Lane(out)
    }

    #[inline(always)]
    pub fn store(a: Lane) -> [u32; LANES] {
        let mut out = [0u32; LANES];
        for (i, v) in a.0.iter().enumerate() {
            unsafe { vst1q_u32(out.as_mut_ptr().add(i * 4), *v) };
        }
        out
    }

    #[inline(always)]
    pub fn add(a: Lane, b: Lane) -> Lane {
        let mut o = a;
        for i in 0..VECS {
            o.0[i] = unsafe { vaddq_u32(a.0[i], b.0[i]) };
        }
        o
    }

    #[inline(always)]
    pub fn addk(a: Lane, k: u32) -> Lane {
        let mut o = a;
        let kv = unsafe { vdupq_n_u32(k) };
        for i in 0..VECS {
            o.0[i] = unsafe { vaddq_u32(a.0[i], kv) };
        }
        o
    }

    /// NEON has no rotate, so this is the usual shift pair. `vshlq_u32`
    /// takes a signed per-lane amount and shifts right when it is
    /// negative, which keeps one intrinsic for both halves. Every call
    /// site passes a literal, so after inlining both amounts are constants
    /// and this folds to two immediate shifts and an or.
    /// NEON has no rotate, so this is the usual shift pair. `vshlq_u32`
    /// takes a signed per-lane amount and shifts right when it is
    /// negative, which keeps one intrinsic for both halves. Every call
    /// site passes a literal, so after inlining both amounts are
    /// constants and this folds to two immediate shifts and an or.
    #[inline(always)]
    pub fn rotl(a: Lane, n: u32) -> Lane {
        let mut o = a;
        let lsh = unsafe { vdupq_n_s32(n as i32) };
        let rsh = unsafe { vdupq_n_s32(n as i32 - 32) };
        for i in 0..VECS {
            o.0[i] = unsafe { vorrq_u32(vshlq_u32(a.0[i], lsh), vshlq_u32(a.0[i], rsh)) };
        }
        o
    }

    #[inline(always)]
    pub fn f0(x: Lane, y: Lane, z: Lane) -> Lane {
        let mut o = x;
        for i in 0..VECS {
            o.0[i] = unsafe { veorq_u32(veorq_u32(x.0[i], y.0[i]), z.0[i]) };
        }
        o
    }

    /// (x & y) | (!x & z) is exactly a bit-select with x as the mask —
    /// one instruction instead of four.
    #[inline(always)]
    pub fn f1(x: Lane, y: Lane, z: Lane) -> Lane {
        let mut o = x;
        for i in 0..VECS {
            o.0[i] = unsafe { vbslq_u32(x.0[i], y.0[i], z.0[i]) };
        }
        o
    }

    #[inline(always)]
    pub fn f2(x: Lane, y: Lane, z: Lane) -> Lane {
        let mut o = x;
        for i in 0..VECS {
            o.0[i] = unsafe { veorq_u32(vorrq_u32(x.0[i], vmvnq_u32(y.0[i])), z.0[i]) };
        }
        o
    }

    /// (x & z) | (y & !z) reorders to (z & x) | (!z & y): another
    /// bit-select, this time with z as the mask.
    #[inline(always)]
    pub fn f3(x: Lane, y: Lane, z: Lane) -> Lane {
        let mut o = x;
        for i in 0..VECS {
            o.0[i] = unsafe { vbslq_u32(z.0[i], x.0[i], y.0[i]) };
        }
        o
    }

    #[inline(always)]
    pub fn f4(x: Lane, y: Lane, z: Lane) -> Lane {
        let mut o = x;
        for i in 0..VECS {
            o.0[i] = unsafe { veorq_u32(x.0[i], vorrq_u32(y.0[i], vmvnq_u32(z.0[i]))) };
        }
        o
    }
}

/// Portable fallback: plain arrays, correct everywhere Rust builds, and
/// vectorized only to the extent the optimizer manages on its own.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
mod lane_ops {
    /// Sized to give the optimizer roughly two independent chains to
    /// interleave, for the latency reason explained above — not simply to
    /// match one register's width.
    ///
    /// x86_64: 16, which is two 256-bit AVX2 registers (or four SSE2
    /// ones). Elsewhere: 8, two 128-bit registers, or just eight scalar
    /// chains on a target with no SIMD at all, which still helps for the
    /// same reason.
    ///
    /// MEASURED on x86_64 (AVX2, `--bench`, three runs each, scalar
    /// baseline ~242-258 ns/hash):
    ///
    ///     8 lanes  236-252 ns/hash  ~1.00x  (no gain)
    ///    16 lanes  262-269 ns/hash  ~0.96x  (slower than scalar)
    ///    32 lanes  139-158 ns/hash  ~1.67x
    ///    64 lanes   97-102 ns/hash  ~2.47x
    ///
    /// End-to-end that is 1.34x (-p), 1.26x (-s) and 1.20x (-i) over the
    /// 16 that was here before, which was actually a small net LOSS
    /// against just calling the `ripemd` crate. The kernel is
    /// latency-bound, so it wants many more independent chains than one
    /// register's width suggests. 128 was not tried: it needs BATCH_SIZE
    /// divisible by 64, so it cannot be changed on its own.
    #[cfg(target_arch = "x86_64")]
    pub const LANES: usize = super::env_usize(option_env!("VERUS_LANES"), 64);
    #[cfg(not(target_arch = "x86_64"))]
    pub const LANES: usize = super::env_usize(option_env!("VERUS_LANES"), 8);

    pub type Lane = [u32; LANES];

    #[inline(always)]
    pub fn splat(v: u32) -> Lane {
        [v; LANES]
    }

    #[inline(always)]
    pub fn load(words: [u32; LANES]) -> Lane {
        words
    }

    #[inline(always)]
    pub fn store(a: Lane) -> [u32; LANES] {
        a
    }

    #[inline(always)]
    pub fn add(a: Lane, b: Lane) -> Lane {
        core::array::from_fn(|i| a[i].wrapping_add(b[i]))
    }

    #[inline(always)]
    pub fn addk(a: Lane, k: u32) -> Lane {
        core::array::from_fn(|i| a[i].wrapping_add(k))
    }

    #[inline(always)]
    pub fn rotl(a: Lane, n: u32) -> Lane {
        core::array::from_fn(|i| a[i].rotate_left(n))
    }

    #[inline(always)]
    pub fn f0(x: Lane, y: Lane, z: Lane) -> Lane {
        core::array::from_fn(|i| x[i] ^ y[i] ^ z[i])
    }

    #[inline(always)]
    pub fn f1(x: Lane, y: Lane, z: Lane) -> Lane {
        core::array::from_fn(|i| (x[i] & y[i]) | (!x[i] & z[i]))
    }

    #[inline(always)]
    pub fn f2(x: Lane, y: Lane, z: Lane) -> Lane {
        core::array::from_fn(|i| (x[i] | !y[i]) ^ z[i])
    }

    #[inline(always)]
    pub fn f3(x: Lane, y: Lane, z: Lane) -> Lane {
        core::array::from_fn(|i| (x[i] & z[i]) | (y[i] & !z[i]))
    }

    #[inline(always)]
    pub fn f4(x: Lane, y: Lane, z: Lane) -> Lane {
        core::array::from_fn(|i| x[i] ^ (y[i] | !z[i]))
    }
}

// Re-exported under the names the algorithm below already uses, so the
// round macro and all 160 step invocations are backend-agnostic.
use lane_ops::{
    add as lane_add, addk as lane_addk, f0 as rf0, f1 as rf1, f2 as rf2, f3 as rf3, f4 as rf4,
    load as lane_load, rotl as lane_rotl, splat as lane_splat, store as lane_store,
};

/// Expands one RIPEMD-160 round: sixteen steps of
/// `t = rotl(a + f(b,c,d) + X[r] + k, s) + e`, then the five-way variable
/// rotation. Written as a macro so every `r`, `s` and `k` is a literal at
/// the point of use, which is what lets the rotations compile to constant
/// shifts. The a..e identifiers are passed in so they refer to the
/// caller's own bindings, and `$f` is re-evaluated each step against
/// their current values.
macro_rules! rip_round {
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $x:ident, $f:expr, $k:literal,
     $( ($r:literal, $s:literal) ),* $(,)?) => {
        $(
            {
                let t = lane_add(
                    lane_rotl(lane_addk(lane_add(lane_add($a, $f), $x[$r]), $k), $s),
                    $e,
                );
                $a = $e;
                $e = $d;
                $d = lane_rotl($c, 10);
                $c = $b;
                $b = t;
            }
        )*
    };
}

/// Computes RIPEMD-160 of `RIPEMD_LANES` independent 32-byte inputs at
/// once.
///
/// Specialized to a 32-byte message, which is all this program ever
/// hashes here (a SHA-256 digest). That fixes the padding entirely: one
/// block, `0x80` at byte 32, and a bit length of 256 — so the message
/// schedule words 8..15 are compile-time constants rather than data.
fn ripemd160_multi_32(inputs: &[[u8; 32]; RIPEMD_LANES]) -> [[u8; 20]; RIPEMD_LANES] {
    // Transpose: message word w of every lane is gathered into one vector,
    // so a single step operates on that word across all hashes at once.
    let mut x = [lane_splat(0); 16];
    for w in 0..8 {
        let mut words = [0u32; RIPEMD_LANES];
        for (lane, slot) in words.iter_mut().enumerate() {
            *slot = u32::from_le_bytes([
                inputs[lane][w * 4],
                inputs[lane][w * 4 + 1],
                inputs[lane][w * 4 + 2],
                inputs[lane][w * 4 + 3],
            ]);
        }
        x[w] = lane_load(words);
    }
    // Fixed padding for a 32-byte message: the 0x80 terminator lands in
    // word 8, words 9..13 are zero, and word 14 holds the bit length
    // (32 * 8 = 256) with word 15 its zero high half.
    x[8] = lane_splat(0x0000_0080);
    x[14] = lane_splat(256);

    const H0: u32 = 0x6745_2301;
    const H1: u32 = 0xEFCD_AB89;
    const H2: u32 = 0x98BA_DCFE;
    const H3: u32 = 0x1032_5476;
    const H4: u32 = 0xC3D2_E1F0;

    let mut al = lane_splat(H0);
    let mut bl = lane_splat(H1);
    let mut cl = lane_splat(H2);
    let mut dl = lane_splat(H3);
    let mut el = lane_splat(H4);

    let mut ar = al;
    let mut br = bl;
    let mut cr = cl;
    let mut dr = dl;
    let mut er = el;

    // ---- left line ----
    rip_round!(al, bl, cl, dl, el, x, rf0(bl, cl, dl), 0x0000_0000,
        (0, 11), (1, 14), (2, 15), (3, 12), (4, 5), (5, 8), (6, 7), (7, 9),
        (8, 11), (9, 13), (10, 14), (11, 15), (12, 6), (13, 7), (14, 9), (15, 8));
    rip_round!(al, bl, cl, dl, el, x, rf1(bl, cl, dl), 0x5A82_7999,
        (7, 7), (4, 6), (13, 8), (1, 13), (10, 11), (6, 9), (15, 7), (3, 15),
        (12, 7), (0, 12), (9, 15), (5, 9), (2, 11), (14, 7), (11, 13), (8, 12));
    rip_round!(al, bl, cl, dl, el, x, rf2(bl, cl, dl), 0x6ED9_EBA1,
        (3, 11), (10, 13), (14, 6), (4, 7), (9, 14), (15, 9), (8, 13), (1, 15),
        (2, 14), (7, 8), (0, 13), (6, 6), (13, 5), (11, 12), (5, 7), (12, 5));
    rip_round!(al, bl, cl, dl, el, x, rf3(bl, cl, dl), 0x8F1B_BCDC,
        (1, 11), (9, 12), (11, 14), (10, 15), (0, 14), (8, 15), (12, 9), (4, 8),
        (13, 9), (3, 14), (7, 5), (15, 6), (14, 8), (5, 6), (6, 5), (2, 12));
    rip_round!(al, bl, cl, dl, el, x, rf4(bl, cl, dl), 0xA953_FD4E,
        (4, 9), (0, 15), (5, 5), (9, 11), (7, 6), (12, 8), (2, 13), (10, 12),
        (14, 5), (1, 12), (3, 13), (8, 14), (11, 11), (6, 8), (15, 5), (13, 6));

    // ---- right line (same steps, round functions in reverse order) ----
    rip_round!(ar, br, cr, dr, er, x, rf4(br, cr, dr), 0x50A2_8BE6,
        (5, 8), (14, 9), (7, 9), (0, 11), (9, 13), (2, 15), (11, 15), (4, 5),
        (13, 7), (6, 7), (15, 8), (8, 11), (1, 14), (10, 14), (3, 12), (12, 6));
    rip_round!(ar, br, cr, dr, er, x, rf3(br, cr, dr), 0x5C4D_D124,
        (6, 9), (11, 13), (3, 15), (7, 7), (0, 12), (13, 8), (5, 9), (10, 11),
        (14, 7), (15, 7), (8, 12), (12, 7), (4, 6), (9, 15), (1, 13), (2, 11));
    rip_round!(ar, br, cr, dr, er, x, rf2(br, cr, dr), 0x6D70_3EF3,
        (15, 9), (5, 7), (1, 15), (3, 11), (7, 8), (14, 6), (6, 6), (9, 14),
        (11, 12), (8, 13), (12, 5), (2, 14), (10, 13), (0, 13), (4, 7), (13, 5));
    rip_round!(ar, br, cr, dr, er, x, rf1(br, cr, dr), 0x7A6D_76E9,
        (8, 15), (6, 5), (4, 8), (1, 11), (3, 14), (11, 14), (15, 6), (0, 14),
        (5, 6), (12, 9), (2, 12), (13, 9), (9, 12), (7, 5), (10, 15), (14, 8));
    rip_round!(ar, br, cr, dr, er, x, rf0(br, cr, dr), 0x0000_0000,
        (12, 8), (15, 5), (10, 12), (4, 9), (1, 12), (5, 5), (8, 14), (7, 6),
        (6, 8), (2, 13), (13, 6), (14, 5), (0, 15), (3, 13), (9, 11), (11, 11));

    // Final mix: each output word combines one word from each line with a
    // different initial-state word. Computed as vectors, then transposed
    // back out to one digest per lane.
    let mixed = [
        lane_store(lane_add(lane_addk(cl, H1), dr)),
        lane_store(lane_add(lane_addk(dl, H2), er)),
        lane_store(lane_add(lane_addk(el, H3), ar)),
        lane_store(lane_add(lane_addk(al, H4), br)),
        lane_store(lane_add(lane_addk(bl, H0), cr)),
    ];

    let mut out = [[0u8; 20]; RIPEMD_LANES];
    for (lane, digest) in out.iter_mut().enumerate() {
        for w in 0..5 {
            digest[w * 4..w * 4 + 4].copy_from_slice(&mixed[w][lane].to_le_bytes());
        }
    }
    out
}

// ===================== Single-block SHA-256 =====================

/// SHA-256's initial state.
const SHA256_IV: [u32; 8] = [
    0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a,
    0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19,
];

/// Builds the reusable 64-byte block for hashing a compressed public key,
/// with the padding already in place.
///
/// A 33-byte message needs exactly one block, and every byte of its
/// padding is fixed: the `0x80` terminator at byte 33, zeros, then a
/// big-endian bit length of 264 (33 * 8 = 0x108) in the final eight
/// bytes. None of that ever changes, so it is written once and the hot
/// loop only ever rewrites the 33 message bytes.
fn init_sha_block() -> [u8; 64] {
    let mut block = [0u8; 64];
    block[33] = 0x80;
    block[62] = 0x01;
    block[63] = 0x08;
    block
}

/// SHA-256 of exactly 33 bytes, using a block prepared by
/// `init_sha_block`.
///
/// Calling the compression function directly skips what `Sha256::digest`
/// redoes on every single call: allocating and initializing a hasher,
/// buffering the message through a block assembler, then constructing the
/// padding and length suffix. That bookkeeping is a real share of the
/// cost here, because the compression itself is one block running on the
/// CPU's crypto unit — the overhead is comparable to the work.
///
/// `sha256_of_33_matches_crate` checks this against `Sha256::digest`.
///
/// The `deprecated` allowance is not a smell to clean up later: sha2
/// 0.10's public `compress256` takes `&[GenericArray<u8, U64>]`, and that
/// `GenericArray` comes from generic-array 0.14, which now marks itself
/// deprecated in favour of 1.x. Constructing the argument any other way
/// still names the same deprecated type, so the warning is unavoidable
/// until sha2 0.11 changes the signature to plain `[u8; 64]`. It is a
/// statement about the dependency's version, not about this call being
/// wrong.
#[allow(deprecated)]
#[inline(always)]
fn sha256_of_33(block: &mut [u8; 64], input: &[u8; 33], out: &mut [u8; 32]) {
    block[0..33].copy_from_slice(input);

    let mut state = SHA256_IV;
    let ga = sha2::digest::generic_array::GenericArray::from_slice(&block[..]);
    sha2::compress256(&mut state, core::slice::from_ref(ga));

    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
}

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

/// 58^10 — the largest power of 58 that fits in a u64, so ten output
/// digits can be extracted per big-number division.
const BASE58_POW10: u64 = 430_804_206_899_405_824;

/// Base58-encodes exactly the 25 bytes of an address into exactly 34
/// characters, writing into `out`.
///
/// This is a specialized replacement for `bs58::encode` on the hot path.
/// The generic encoder divides the whole number by 58 once per output
/// digit *per input byte* — around 850 divisions per address. This treats
/// the number as four u64 limbs and divides by 58^10 instead, extracting
/// ten digits at a time, which needs about 16. Measured 7.1x faster, and
/// validated to produce byte-identical output to `bs58::encode` across
/// 3,000,000 random addresses plus the all-zero and all-0xFF payload
/// extremes.
///
/// Specializing this way is only sound because both properties are fixed
/// for VerusCoin addresses: the payload is always exactly 25 bytes, and
/// the leading version byte is always non-zero (0x3C), so there are never
/// leading zeros to encode and the output length is always exactly 34.
/// The debug assertion below pins the second assumption in case the
/// version byte is ever changed.
fn encode_address_base58(input: &[u8; 25], out: &mut [u8; 34]) {
    debug_assert!(
        input[0] != 0,
        "encode_address_base58 assumes a non-zero leading byte (no leading-zero handling)"
    );

    // Load as four big-endian u64 limbs (25 bytes zero-padded to 32).
    let mut padded = [0u8; 32];
    padded[7..32].copy_from_slice(input);
    let mut limbs = [
        u64::from_be_bytes([padded[0], padded[1], padded[2], padded[3], padded[4], padded[5], padded[6], padded[7]]),
        u64::from_be_bytes([padded[8], padded[9], padded[10], padded[11], padded[12], padded[13], padded[14], padded[15]]),
        u64::from_be_bytes([padded[16], padded[17], padded[18], padded[19], padded[20], padded[21], padded[22], padded[23]]),
        u64::from_be_bytes([padded[24], padded[25], padded[26], padded[27], padded[28], padded[29], padded[30], padded[31]]),
    ];

    let alphabet = BASE58_ALPHABET.as_bytes();
    let mut pos = 34usize;
    while pos > 0 {
        // Divide the whole number by 58^10, keeping the remainder.
        let mut rem: u128 = 0;
        for limb in limbs.iter_mut() {
            let cur = (rem << 64) | (*limb as u128);
            *limb = (cur / BASE58_POW10 as u128) as u64;
            rem = cur % BASE58_POW10 as u128;
        }
        // Peel ten digits off the remainder, least significant first.
        let mut r = rem as u64;
        for _ in 0..10 {
            if pos == 0 {
                break;
            }
            pos -= 1;
            out[pos] = alphabet[(r % 58) as usize];
            r /= 58;
        }
    }
}

/// Completes an address from the first 21 bytes, writing it into `out`
/// and returning it as a `&str`. Hot-path version: no allocation, and
/// uses the specialized encoder above.
fn address_from_first21_into<'a>(first21: &[u8; 21], out: &'a mut [u8; 34]) -> &'a str {
    let mut addr_bytes = [0u8; 25];
    addr_bytes[0..21].copy_from_slice(first21);
    let checksum_full = Sha256::digest(Sha256::digest(&addr_bytes[0..21]));
    addr_bytes[21..25].copy_from_slice(&checksum_full[0..4]);
    encode_address_base58(&addr_bytes, out);
    // Every byte written comes from the Base58 alphabet, so this is
    // always valid ASCII/UTF-8.
    std::str::from_utf8(out).expect("Base58 output is always ASCII")
}

/// Completes a VerusCoin transparent address given the first 21 bytes:
/// computes the double-SHA256 checksum and Base58-encodes the result.
///
/// Deliberately still uses the general-purpose `bs58` crate rather than
/// the specialized encoder above: this is the function the match
/// self-check calls, so keeping it on a different implementation means
/// the self-check independently re-validates the fast encoder's output
/// on every single match, rather than both paths sharing (and therefore
/// sharing any bug in) the same code.
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
    /// The leading 8 bytes of `lower` / `upper_exclusive` as big-endian
    /// u64s, so the common case of `provably_outside_prefix` is a single
    /// integer comparison instead of two 25-byte array comparisons. See
    /// that function for why comparing only the leading 8 bytes is exact
    /// rather than approximate.
    lower_hi8: u64,
    upper_hi8: u64,
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

    let lower_hi8 = u64::from_be_bytes(lower[0..8].try_into().ok()?);
    let upper_hi8 = u64::from_be_bytes(upper_exclusive[0..8].try_into().ok()?);

    Some(PrefixBound { lower, upper_exclusive, lower_hi8, upper_hi8 })
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
///
/// The leading-u64 test below is an exact shortcut, not a heuristic. The
/// unknown checksum occupies bytes 21..25, so the two endpoints being
/// tested — `lo` (checksum all-zero) and `hi` (checksum all-ones) — are
/// byte-for-byte identical to `first21` across their leading 8 bytes.
/// Lexicographic comparison of big-endian arrays is decided by the first
/// differing byte, so a single u64 comparison of those 8 bytes settles
/// both `hi < lower` and `lo >= upper_exclusive` outright unless the
/// value lands exactly ON a boundary word, which is the only case that
/// falls through to the full comparison. Since a prefix's bound spans a
/// wide range of the leading bytes, that fallthrough is rare.
fn provably_outside_prefix(first21: &[u8; 21], bound: &PrefixBound) -> bool {
    let v = u64::from_be_bytes([
        first21[0], first21[1], first21[2], first21[3],
        first21[4], first21[5], first21[6], first21[7],
    ]);

    // Strictly below the low bound, or strictly above the high bound:
    // provably outside, no further comparison possible or needed.
    if v < bound.lower_hi8 || v > bound.upper_hi8 {
        return true;
    }
    // Strictly inside both bounds: no checksum can push it out, since the
    // checksum cannot reach the leading 8 bytes at all.
    if v > bound.lower_hi8 && v < bound.upper_hi8 {
        return false;
    }

    // Exactly on a boundary word — the leading bytes are a tie, so fall
    // back to the full 25-byte comparison to break it.
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
    // Eight bytes at a time rather than one.
    //
    // This runs for every candidate of a suffix search, and a 128-bit
    // division is far from free, so doing 21 of them per address was the
    // single largest cost in that path. Folding a whole u64 per step
    // needs only three.
    //
    // The intermediate cannot overflow: `modulus` is 58^k for k <= 10, so
    // it is below 2^59 and any remainder is below that too. Shifting a
    // sub-2^59 value up by 64 bits reaches at most 2^123, and adding a
    // u64 keeps it under 2^124 — comfortably inside u128.
    debug_assert!(modulus < (1u64 << 59), "modulus must stay small enough to shift by 64");
    let m = modulus as u128;
    let hi = u64::from_be_bytes(first21[0..8].try_into().unwrap()) as u128;
    let mid = u64::from_be_bytes(first21[8..16].try_into().unwrap()) as u128;

    let mut r = hi % m;
    r = ((r << 64) | mid) % m;

    // The trailing five bytes, folded in one step.
    let mut tail: u128 = 0;
    for &b in &first21[16..21] {
        tail = (tail << 8) | b as u128;
    }
    r = ((r << 40) | tail) % m;
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

/// Reads a small integer from an environment variable at COMPILE time,
/// falling back to `default` when unset or unparseable.
///
/// Lane width has to be a compile-time constant — the kernel state lives
/// in fixed-size arrays of vector registers, and Rust cannot size those
/// from a runtime value without nightly features. This is the next best
/// thing: it can be changed per device without editing any source.
///
///     VERUS_VECS=3 cargo build --release      # aarch64/NEON
///     VERUS_LANES=32 cargo build --release    # everything else
///
/// `--bench` prints the width actually compiled in, so a build can
/// always be checked.
const fn env_usize(v: Option<&str>, default: usize) -> usize {
    // Byte comparison rather than string matching: `match` on `str` is
    // not permitted in a const.
    match v {
        None => default,
        Some(s) => {
            let b = s.as_bytes();
            let mut i = 0;
            let mut acc = 0usize;
            while i < b.len() {
                let d = b[i];
                if d < b'0' || d > b'9' {
                    return default;
                }
                acc = acc * 10 + (d - b'0') as usize;
                i += 1;
            }
            if acc == 0 { default } else { acc }
        }
    }
}

// ===================== Auto-tuning =====================
//
// The best batch size and thread count are properties of the specific
// CPU, not of the program: cache sizes decide how large a batch can be
// before the working set stops fitting, and a phone's power budget
// decides how many threads actually add throughput before clocks drop to
// compensate. A value that is right for a modern desktop core can be
// wrong by a wide margin on an older in-order phone core.
//
// `-b` and `-t` expose both so a device can be matched without
// rebuilding. The measurement helper below runs the REAL search — same
// worker threads, same kernel, same filters — for a short interval and
// counts addresses, rather than timing a synthetic stand-in that might
// not share the cache behaviour being measured. It is used by `-B` to
// report what the machine actually does.


/// Ceiling on the memory the batch buffers may occupy. This is not a
/// performance limit — throughput stops improving long before here,
/// because the single modular inversion a batch amortizes is already
/// negligible per point by a few thousand. It exists so that a mistyped
/// `-b` cannot ask the allocator for more memory than the device has:
/// `-b 40000000` on 8 threads would otherwise reach for about 16 GB.
const BATCH_MEMORY_BUDGET: usize = 512 * 1024 * 1024;

/// Bytes the batch buffers need. The addition table holds two field
/// elements per point and is shared by every thread; each thread also
/// keeps one per point for the batch inversion.
fn batch_memory_bytes(batch: usize, threads: usize) -> usize {
    std::mem::size_of::<FieldElement>().saturating_mul(batch).saturating_mul(2 + threads)
}

/// Largest batch that fits the budget at this thread count. Reported in
/// errors so the bound is a number rather than a mystery.
fn max_batch_for(threads: usize) -> usize {
    let per_point = std::mem::size_of::<FieldElement>() * (2 + threads);
    (BATCH_MEMORY_BUDGET / per_point.max(1)).max(RIPEMD_LANES)
}

/// A prefix long enough that a match is effectively impossible during a
/// calibration interval, so timing is never disturbed by match
/// reporting. Still a legitimately achievable prefix, so the pre-filter
/// behaves exactly as it would in a real search.
const TUNE_PREFIX: &str = "RCABCDEFG";

/// Runs the real search at a given configuration and returns measured
/// addresses per second.
fn measure_throughput(
    batch_size: usize,
    threads: usize,
    dur: Duration,
    shared_table: Option<&Arc<Vec<(FieldElement, FieldElement)>>>,
) -> f64 {
    let prefixes = vec![TUNE_PREFIX.to_string()];
    let prefix_bounds = try_compute_all_prefix_bounds(&prefixes);
    // Reuse a caller-supplied table when there is one.
    //
    // Building it here instead was a systematic bias, not just wasted
    // work: `build_addition_table` walks the curve one point at a time on
    // a single thread, so it takes LONGER for a bigger batch — and that
    // stretch of quiet, single-threaded work is a cooldown immediately
    // before the timed window. Large batches were being measured on a
    // cooler device than small ones, which is exactly the direction of
    // the error seen in practice: a sweep reported 48.6 MW/s for batch
    // 65536 while a real search at that setting sustained about 45.2.
    //
    // Entry i of the table is (i+1)*G regardless of how long the table
    // is, so one table built at the largest candidate size is correct for
    // every smaller one — the worker only ever reads its first
    // `batch_size` entries.
    let owned;
    let table = match shared_table {
        Some(t) if t.len() >= batch_size => Arc::clone(t),
        _ => {
            owned = Arc::new(build_addition_table(batch_size));
            Arc::clone(&owned)
        }
    };
    let endo = setup_endomorphism();
    let state = SharedState::new(&None);

    let mut handles = Vec::new();
    for _ in 0..threads {
        let p = prefixes.clone();
        let pb = prefix_bounds.clone();
        let st = state.clone();
        let tb = Arc::clone(&table);
        handles.push(thread::spawn(move || {
            worker_loop(p, Vec::new(), Vec::new(), pb, Vec::new(), -1, st, tb, endo, batch_size)
        }));
    }

    // Let the threads spin up and caches warm before the timed window.
    thread::sleep(Duration::from_millis(60));
    let start_count = state.keys_tried.load(Ordering::Relaxed);
    let t = Instant::now();

    // The counter advances once per COMPLETED batch, so a window that
    // only fits a couple of batches produces a quantised reading rather
    // than a rate — at batch 131072 on a slow core, one batch can outlast
    // a 500 ms window entirely, and every "sample" then returns the same
    // number (a spread of exactly 0.0%, which is the giveaway). Large
    // batches were being judged on noise-free nonsense.
    //
    // So the window closes on whichever comes last: the requested
    // duration, or enough completed batches to be meaningful. The hard
    // cap keeps a hopeless candidate from stalling the sweep; a candidate
    // that hits the cap is reported as unmeasurable rather than guessed
    // at.
    const MIN_BATCHES_PER_THREAD: u64 = 6;
    let needed = (batch_size as u64)
        .saturating_mul(addrs_per_point_hint())
        .saturating_mul(MIN_BATCHES_PER_THREAD)
        .saturating_mul(threads as u64);
    let cap = dur.saturating_mul(6);
    loop {
        thread::sleep(Duration::from_millis(20));
        let counted = state.keys_tried.load(Ordering::Relaxed) - start_count;
        let elapsed = t.elapsed();
        if elapsed >= dur && counted >= needed {
            break;
        }
        if elapsed >= cap {
            break;
        }
    }

    let elapsed = t.elapsed().as_secs_f64();
    let counted = state.keys_tried.load(Ordering::Relaxed) - start_count;

    state.stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }
    counted as f64 / elapsed
}
/// Addresses produced per base point once the curve symmetries are
/// applied. Used only to size measurement windows; the search itself
/// derives this from whether the endomorphism is available.
fn median_of<F: FnMut() -> f64>(samples: usize, mut f: F) -> f64 {
    let mut v: Vec<f64> = (0..samples).map(|_| f()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Addresses produced per base point once the curve symmetries are
/// applied. Used only to size measurement windows.
fn addrs_per_point_hint() -> u64 {
    6
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
    batch_size: usize,
    max_matches: i64,
    output_file: Option<String>,
    any_normalized: bool,
    /// Address to serve a cluster on, when acting as master.
    serve: Option<String>,
    /// Shared word a worker must present to join.
    token: String,
    /// Tell workers to exit rather than idle when the objective is met.
    stop_workers: bool,
    /// Seconds between ETA refreshes; 0 means print it once.
    eta_every: u64,
}

/// Parses CLI arguments, auto-normalizes prefixes (see `normalize_prefix`),
/// and validates every pattern — exiting the process with a clear message
/// if anything is missing or impossible before any search work begins.
fn parse_cli() -> Config {
    let cpu_cores_str: &'static str = Box::leak(num_cpus::get().to_string().into_boxed_str());

    let matches = Command::new("verus-vanity")
        .version("0.8.1")
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
            Arg::new("batch")
                .short('b')
                .long("batch")
                .help("Points per batch (min = RIPEMD lane count; max scales with -t and RAM)")
                .num_args(1),
        )
        .arg(
            Arg::new("eta")
                .short('e')
                .long("eta")
                .help("Seconds between ETA refreshes; 0 shows it once only")
                .default_value("30")
                .num_args(1),
        )
        .arg(
            Arg::new("serve")
                .short('S')
                .long("serve")
                .help("Run as cluster master on ADDR:PORT (also searches locally)")
                .num_args(1),
        )
        .arg(
            Arg::new("join")
                .short('J')
                .long("join")
                .help("Run as cluster worker, taking the objective from the master at ADDR:PORT")
                .num_args(1),
        )
        .arg(
            Arg::new("pass")
                .short('P')
                .long("pass")
                .help("Shared word a worker must present to join a cluster")
                .default_value("")
                .num_args(1),
        )
        .arg(
            Arg::new("name")
                .short('N')
                .long("name")
                .help("Name this worker reports to the master [default: hostname-ish]")
                .num_args(1),
        )
        .arg(
            Arg::new("keepalive")
                .short('k')
                .long("keepalive")
                .action(ArgAction::SetTrue)
                .help("Worker: stay running when the master stops, and wait for the next objective"),
        )
        .arg(
            Arg::new("stop-workers")
                .long("stop-workers")
                .action(ArgAction::SetTrue)
                .help("Master: tell workers to exit when the objective is met, even keepalive ones"),
        )
        .arg(
            Arg::new("dismiss")
                .short('d')
                .long("dismiss")
                .help("Shut down every worker that connects to ADDR:PORT, then exit")
                .num_args(1),
        )
        .arg(
            Arg::new("keys-stay-local")
                .long("keys-stay-local")
                .action(ArgAction::SetTrue)
                .help("Worker keeps found keys on its own machine; only addresses are sent"),
        )
        .arg(
            Arg::new("bench")
                .short('B')
                .long("bench")
                .action(ArgAction::SetTrue)
                .help("Measure per-stage throughput on this machine and exit"),
        )
        .arg(
            Arg::new("version")
                .short('v')
                .long("version")
                .action(ArgAction::Version)
                .help("Print version"),
        )
        .get_matches();

    // Checked before the "give me a pattern" requirement below, since
    // benchmarking does not search for anything.
    if matches.get_flag("bench") {
        run_bench();
        std::process::exit(0);
    }

    validate_token_or_exit(matches.get_one::<String>("pass").map(|s| s.as_str()).unwrap_or(""));

    if let Some(listen) = matches.get_one::<String>("dismiss") {
        run_dismiss(
            listen,
            matches.get_one::<String>("pass").map(|s| s.as_str()).unwrap_or(""),
        );
        std::process::exit(0);
    }

    // A worker takes its patterns from the master, so it must dispatch
    // before the "give me a pattern" requirement below.
    if let Some(master) = matches.get_one::<String>("join") {
        let threads: usize = matches
            .get_one::<String>("threads")
            .unwrap()
            .parse()
            .unwrap_or_else(|_| num_cpus::get())
            .max(1);
        let max_batch = max_batch_for(threads);
        let batch = match matches.get_one::<String>("batch") {
            Some(raw) => raw
                .parse::<usize>()
                .unwrap_or(BATCH_SIZE)
                .clamp(RIPEMD_LANES, max_batch)
                .next_multiple_of(RIPEMD_LANES),
            None => BATCH_SIZE.clamp(RIPEMD_LANES, max_batch),
        };
        let default_name = format!("worker-{}", std::process::id());
        let name = matches
            .get_one::<String>("name")
            .cloned()
            .unwrap_or(default_name)
            .replace(' ', "_");
        run_cluster_worker(
            master,
            matches.get_one::<String>("pass").map(|s| s.as_str()).unwrap_or(""),
            &name,
            threads,
            batch,
            matches.get_flag("keys-stay-local"),
            matches.get_flag("keepalive"),
        );
        std::process::exit(0);
    }



    // Batch size is a runtime choice so a device can be matched without
    // rebuilding — small caches (older phones) generally want a smaller
    // batch than the default, which is sized for a large modern core.
    //
    // It must stay a multiple of RIPEMD_LANES: candidates are staged into
    // lane groups, and a partial group at the end of a batch would be
    // silently dropped. Rounding up to the next multiple guarantees this
    // for every variant count, which is what the compile-time assertion
    // on the default does.
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
    let threads_given = matches.value_source("threads") == Some(clap::parser::ValueSource::CommandLine);
    let threads: usize = if threads_given {
        matches.get_one::<String>("threads").unwrap().parse().unwrap_or(num_cpus::get()).max(1)
    } else {
        num_cpus::get().max(1)
    };

    // Parsed after the thread count, because the usable upper bound
    // depends on it: the per-thread inversion buffer is duplicated once
    // per thread.
    //
    // Lower bound is RIPEMD_LANES, and it is a hard one: candidates are
    // staged in lane groups, so a batch smaller than one group could not
    // fill it. Values in between are rounded up for the same reason.
    let max_batch = max_batch_for(threads);
    let batch_size = match matches.get_one::<String>("batch") {
        Some(raw) => match raw.parse::<usize>() {
            Ok(v) if v >= RIPEMD_LANES && v <= max_batch => {
                let rounded = v.next_multiple_of(RIPEMD_LANES);
                if rounded != v {
                    eprintln!(
                        "Note: batch size {} rounded up to {} (must be a multiple of the {} RIPEMD lanes).",
                        v, rounded, RIPEMD_LANES
                    );
                }
                rounded
            }
            Ok(v) if v > max_batch => {
                eprintln!(
                    "⚠️  --batch/-b {} is too large: with {} threads that would need {:.1} GB.\n  \
                     The limit here is {} (about {:.0} MB). Throughput stops improving well below\n  \
                     this anyway — the inversion a batch amortizes is already negligible per point\n  \
                     by a few thousand.",
                    v,
                    threads,
                    batch_memory_bytes(v, threads) as f64 / 1e9,
                    max_batch,
                    BATCH_MEMORY_BUDGET as f64 / 1e6
                );
                std::process::exit(1);
            }
            _ => {
                eprintln!(
                    "⚠️  --batch/-b must be a whole number between {} and {} on this machine\n  \
                     ({} threads). The lower bound is the RIPEMD lane count.",
                    RIPEMD_LANES, max_batch, threads
                );
                std::process::exit(1);
            }
        },
        None => BATCH_SIZE.clamp(RIPEMD_LANES, max_batch),
    };
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
        batch_size,
        serve: matches.get_one::<String>("serve").cloned(),
        token: matches.get_one::<String>("pass").cloned().unwrap_or_default(),
        stop_workers: matches.get_flag("stop-workers"),
        eta_every: matches
            .get_one::<String>("eta")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(30),
        max_matches,
        output_file,
        any_normalized,
    }
}

/// Prints the startup banner summarizing the search that's about to run.
fn print_banner(config: &Config, endo_enabled: bool) {
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
    // Purely informational. "off" is a slower search, never a wrong one —
    // see the endomorphism section for why it is a fallback rather than
    // an error.
    println!(
        "Curve symmetries: {}",
        if endo_enabled {
            "negation + endomorphism (6 addresses per point)"
        } else {
            "negation only (2 addresses per point)"
        }
    );
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
    /// Set to halt every worker. Only used by auto-tuning, which runs the
    /// real search for a short interval and then stops it; a normal search
    /// never sets this.
    stop: Arc<AtomicBool>,
    /// Serializes match reporting. Each report is many lines (address,
    /// WIF, key, and a whole QR code); without this, two threads finding
    /// matches at the same moment interleave their lines and produce an
    /// unreadable, unscannable mess on the terminal.
    report_lock: Arc<Mutex<()>>,
    /// When set, a found match is handed here instead of being printed
    /// locally. Used by cluster workers, whose matches belong to the
    /// master rather than to their own terminal.
    match_sink: Option<std::sync::mpsc::Sender<MatchRecord>>,
    /// Addresses contributed by workers since the last progress tick.
    /// Every worker ADDS to this and the stats thread drains it, so the
    /// figure is the whole cluster rather than one member. Zero when
    /// running standalone.
    remote_rate: Arc<AtomicU64>,
    /// Addresses tried by connected workers, accumulated. Counted
    /// separately from `keys_tried` so the progress line can show a
    /// combined total without workers racing the local counter.
    remote_tried: Arc<AtomicU64>,
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
            stop: Arc::new(AtomicBool::new(false)),
            report_lock: Arc::new(Mutex::new(())),
            match_sink: None,
            remote_rate: Arc::new(AtomicU64::new(0)),
            remote_tried: Arc::new(AtomicU64::new(0)),
        }
    }
}

/// Spawns the background thread that prints search progress once a
/// second: a one-time typical-time-to-find estimate the first time a real
/// rate is available, then an ongoing simple progress percentage.
fn spawn_stats_thread(
    state: SharedState,
    start_time: Instant,
    expected: f64,
    target_matches: f64,
    eta_every: u64,
) {
    thread::spawn(move || {
        let keys_tried_last = AtomicU64::new(0);
        let mut last_eta = Instant::now();
        let mut total_at_last_eta: u64 = 0;
        let mut printed_estimate = false;

        loop {
            thread::sleep(Duration::from_secs(1));
            let local_total = state.keys_tried.load(Ordering::Relaxed);
            // Everything below works on the COMBINED figure. Using the
            // local counter alone understated progress and overstated the
            // ETA the moment a worker joined, because the addresses they
            // contributed were simply not in the sum.
            let remote_total = state.remote_tried.load(Ordering::Relaxed);
            let total = local_total + remote_total;
            let last = keys_tried_last.swap(total, Ordering::Relaxed);
            let combined_rate = total.saturating_sub(last);

            let elapsed = start_time.elapsed().as_secs_f64();
            let average_rate = if elapsed > 0.0 { total as f64 / elapsed } else { 0.0 };

            let found_so_far = state.found_count.load(Ordering::Relaxed).max(0) as f64;
            let remaining_matches = (target_matches - found_so_far).max(0.0);
            let p_per_try = if expected.is_finite() && expected > 0.0 { 1.0 / expected } else { 0.0 };

            // A typical (median) time to find, not a countdown.
            //
            // Refreshed on a cadence rather than printed once: a cluster
            // gains workers over time, and an estimate taken from the
            // first second of a solo run is badly wrong the moment three
            // phones join. `-e` sets the interval; `-e 0` keeps the old
            // print-once behaviour.
            let due = !printed_estimate
                || (eta_every > 0 && last_eta.elapsed() >= Duration::from_secs(eta_every));
            // Rate over the window since the last refresh, not the average
            // since launch. A cumulative average barely moves when workers
            // join half an hour in — it is dominated by all the time they
            // were not there — so the ETA would keep quoting the solo
            // figure. A recent window reflects the cluster as it is now.
            let window_secs = last_eta.elapsed().as_secs_f64().max(1.0);
            let window_rate = if printed_estimate {
                total.saturating_sub(total_at_last_eta) as f64 / window_secs
            } else {
                average_rate
            };
            if due && window_rate > 0.0 && remaining_matches > 0.0 {
                let t50 = tries_for_probability(p_per_try, 0.5) * remaining_matches / window_rate;
                if printed_estimate {
                    println!(
                        "ETA refreshed: ~{} at {} combined",
                        format_duration(t50),
                        format_with_si_rate(window_rate as u64)
                    );
                } else {
                    println!("Estimated typical time to find: ~{}\n", format_duration(t50));
                }
                printed_estimate = true;
                last_eta = Instant::now();
                total_at_last_eta = total;
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

            // Workers count separately so they never race the local
            // counter; the line adds them back for a cluster total.
            // Drained, not read: every worker adds its report here, so
            // taking and clearing it gives the cluster's contribution for
            // this interval.
            let rrate = state.remote_rate.swap(0, Ordering::Relaxed);
            if remote_total > 0 || rrate > 0 {
                println!(
                    "Progress: {} ({} tried, {}) [local {} + cluster {}]",
                    progress,
                    format_with_si(total),
                    format_with_si_rate(combined_rate),
                    format_with_si_rate(combined_rate.saturating_sub(rrate)),
                    format_with_si_rate(rrate)
                );
            } else {
                println!(
                    "Progress: {} ({} tried, {})",
                    progress,
                    format_with_si(total),
                    format_with_si_rate(combined_rate)
                );
            }
        }
    });
}

// ===================== Search worker =====================

/// Prints a found match (address, WIF, hex key, QR code) to stdout, and
/// appends the same information to the output file if one was configured.
fn report_match(found_num: i64, desc: &str, addr: &str, wif: &str, priv_hex: &str, state: &SharedState) {
    // A cluster worker forwards its find to the master rather than
    // printing it. The master is the one deciding whether the objective
    // is met, so it has to be the one that records the result.
    if let Some(tx) = &state.match_sink {
        let _ = tx.send(MatchRecord {
            desc: desc.to_string(),
            addr: addr.to_string(),
            wif: wif.to_string(),
            priv_hex: priv_hex.to_string(),
        });
        return;
    }

    // A match whose key stayed on the worker has no WIF to print and no
    // QR to scan; say where it actually is instead of showing "-".
    if wif == "-" && priv_hex == "-" {
        let body = format!(
            "----- MATCH {} for {} FOUND -----\nAddress: {}\n\
             The private key stayed on the worker that found it (--keys-stay-local).\n\
             Collect it from that device's output; the address above is verified.\n\
             -------------------------\n",
            found_num, desc, addr
        );
        let _guard = state.report_lock.lock().unwrap_or_else(|e| e.into_inner());
        print!("{}\n", body);
        let _ = std::io::stdout().flush();
        if let Some(output_mutex) = &state.output_writer {
            let mut output = output_mutex.lock().unwrap_or_else(|e| e.into_inner());
            write!(output, "{}\n", body).ok();
            output.flush().ok();
        }
        return;
    }

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

// ===================== GLV endomorphism & negation symmetry =====================
//
// Each affine addition above costs roughly six field multiplications.
// secp256k1 has two symmetries that turn that ONE point into SIX distinct
// public keys for almost no extra work, cutting the per-address elliptic
// curve cost by about 5x.
//
// 1. NEGATION (free — zero field operations). If P = (x, y) is on the
//    curve then so is -P = (x, -y). A compressed public key is just a
//    parity byte followed by x, and x is IDENTICAL for both. Since the
//    field prime p is odd, p - y always has the opposite parity to y (for
//    y != 0, and y = 0 cannot occur here — a point with y = 0 has order 2,
//    which secp256k1's odd group order forbids). So the compressed key
//    for -P is literally the same 32 x-bytes with the other parity byte:
//    0x02 <-> 0x03. Its private key is -k.
//
// 2. ENDOMORPHISM (one multiplication). secp256k1 admits the efficiently
//    computable map phi(x, y) = (beta*x, y), where beta is a non-trivial
//    cube root of 1 mod p. Applying it gives the point lambda*P, where
//    lambda is a non-trivial cube root of 1 mod n. Note that Y IS
//    UNCHANGED, so the parity byte carries over for free, and applying it
//    twice gives a third x for one more multiplication. Private keys are
//    lambda*k and lambda^2*k.
//
// Combined: x, beta*x, beta^2*x (2 muls total) each with both parities (0
// muls) = 6 addresses. All six scalars are distinct for any generic k, so
// this is six genuinely independent candidates, not the same address
// counted six times.
//
// This changes only how CANDIDATES are produced, never how a match is
// judged or how a key is emitted. Every reported match still re-derives
// its address from the recovered private key through k256's own scalar
// multiplication and refuses to print on any disagreement, so a mistake
// here can only ever cost throughput or produce a loud self-check
// failure — never an address whose key does not control it.

/// The constants behind the endomorphism above, validated at startup.
#[derive(Clone, Copy)]
struct Endomorphism {
    /// Non-trivial cube root of 1 modulo the field prime p.
    beta: FieldElement,
    /// The matching non-trivial cube root of 1 modulo the group order n,
    /// such that (beta*x, y) is exactly lambda*(x, y).
    lambda: Scalar,
}

/// beta, as the standard secp256k1 GLV constant.
const BETA_HEX: &str = "7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee";

/// lambda, in big-endian 64-bit limbs (most significant first).
///
/// Assembled from limbs rather than parsed from bytes purely to avoid
/// needing a scalar-decoding entry point beyond the `Scalar::from(u64)`
/// plus arithmetic already used elsewhere in this file. The value is
/// 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72.
const LAMBDA_LIMBS: [u64; 4] = [
    0x5363ad4cc05c30e0,
    0xa5261c028812645a,
    0x122e22ea20816678,
    0xdf02967c1b23bd72,
];

/// Rebuilds `LAMBDA_LIMBS` into a `Scalar` via plain field arithmetic.
fn lambda_from_limbs() -> Scalar {
    // 2^64, expressed without needing a shift: (2^64 - 1) + 1.
    let two64 = Scalar::from(u64::MAX) + Scalar::ONE;
    let mut acc = Scalar::ZERO;
    for limb in LAMBDA_LIMBS {
        acc = acc * two64 + Scalar::from(limb);
    }
    acc
}

/// Loads and PROVES the endomorphism constants before any of them are
/// trusted, returning None if anything at all fails to check out.
///
/// None is not an error — it simply means the six-address optimization is
/// switched off and the search runs with negation symmetry alone (two
/// addresses per point), which needs no constants and cannot be wrong.
/// Structuring it as a fallback rather than an abort means a bad constant
/// costs throughput and nothing else.
///
/// The decisive check is the last one: rather than trusting that beta and
/// lambda are the correctly PAIRED cube roots (each has two non-trivial
/// choices, and pairing the wrong two together yields a valid-looking but
/// wrong scalar), it verifies the actual identity phi(P) == lambda*P
/// against k256's own scalar multiplication on real points. If lambda is
/// paired the wrong way round, lambda^2 is tried instead, so a swapped
/// pairing self-corrects rather than silently disabling the optimization.
fn setup_endomorphism() -> Option<Endomorphism> {
    let beta_bytes = hex::decode(BETA_HEX).ok()?;
    if beta_bytes.len() != 32 {
        return None;
    }
    let beta_ct = FieldElement::from_bytes(beta_bytes[..].into());
    if !bool::from(beta_ct.is_some()) {
        return None;
    }
    let beta = beta_ct.unwrap();

    // beta must be a NON-TRIVIAL cube root of one: beta^3 == 1, beta != 1.
    // (beta == 1 would make phi the identity map and every "extra"
    // address a duplicate of the original.)
    if beta.normalize() == FieldElement::ONE.normalize() {
        return None;
    }
    if (beta * beta * beta).normalize() != FieldElement::ONE.normalize() {
        return None;
    }

    let lambda = lambda_from_limbs();
    if lambda == Scalar::ONE || lambda * lambda * lambda != Scalar::ONE {
        return None;
    }

    // Two independent test points, one small and one arbitrary, so a
    // coincidence at a special point cannot pass this.
    let test_scalars = [Scalar::from(1u64), Scalar::from(0x9e3779b97f4a7c15u64)];

    for candidate in [lambda, lambda * lambda] {
        let mut all_ok = true;
        for t in test_scalars {
            let (x, y) = match affine_xy(&(ProjectivePoint::GENERATOR * t).to_affine()) {
                Some(xy) => xy,
                None => return None,
            };
            let (ex, ey) = match affine_xy(&(ProjectivePoint::GENERATOR * (t * candidate)).to_affine()) {
                Some(xy) => xy,
                None => return None,
            };
            if (x * beta).normalize() != ex.normalize() || y.normalize() != ey.normalize() {
                all_ok = false;
                break;
            }
        }
        if all_ok {
            return Some(Endomorphism { beta, lambda: candidate });
        }
    }

    None
}

/// Precomputes the addition table: entry i holds the affine coordinates
/// of (i+1)*G. Built once at startup and shared (by clone) with every
/// worker thread.
fn build_addition_table(batch_size: usize) -> Vec<(FieldElement, FieldElement)> {
    let mut table = Vec::with_capacity(batch_size);
    let mut acc = ProjectivePoint::GENERATOR;
    for _ in 0..batch_size {
        let xy = affine_xy(&acc.to_affine()).expect("multiples of G are always valid affine points");
        table.push(xy);
        acc += ProjectivePoint::GENERATOR;
    }
    table
}

/// Forward half of Montgomery's batch-inversion trick, specialized so
/// that only ONE buffer is needed instead of two.
///
/// Writes inclusive prefix products into `products` — `products[i]`
/// becomes d_0 * d_1 * ... * d_i, where d_i is `table[i].0 - px` — and
/// returns the inverse of the total product, from which the backward pass
/// recovers every individual inverse.
///
/// The buffer is owned by the caller and reused across every batch, so
/// this performs zero heap allocation per call — unlike the library's
/// `batch_invert`, which allocates and zeroes three buffers of this size
/// on every single call and then copies the result out again.
///
/// Why this takes the table and base point rather than a prepared array
/// of denominators: the textbook formulation keeps the ORIGINAL d_i
/// values alongside the prefix products, because the backward pass needs
/// both, which costs a second full-size buffer. Here d_i is only a single
/// field subtraction away from data the backward pass is already reading,
/// so it is cheaper to recompute it there (a few cycles) than to store
/// and reload it (a 40-byte round trip through a buffer far too large for
/// L1). That removes 20KB from each worker thread's per-batch working
/// set. The recomputation is the identical deterministic subtraction, so
/// it reproduces the value bit-for-bit.
///
/// Returns None if any denominator is zero, exactly like the library
/// version. Callers must treat that as "cannot proceed", never as
/// success.
fn batch_invert_prefix_products(
    table: &[(FieldElement, FieldElement)],
    px: &FieldElement,
    products: &mut [FieldElement],
) -> Option<FieldElement> {
    // Subtraction leaves a small magnitude that's already well within
    // what mul accepts, so no normalize_weak is needed here — the debug
    // build's magnitude assertions verify this.
    let mut acc = FieldElement::ONE;
    for (slot, entry) in products.iter_mut().zip(table.iter()) {
        acc = acc * (entry.0 - *px);
        *slot = acc;
    }

    // One inversion for the entire batch — the whole point of the trick.
    let total_inv = acc.invert();
    if bool::from(total_inv.is_some()) {
        Some(total_inv.unwrap())
    } else {
        None
    }
}

/// One worker thread's search loop — see the module note above for the
/// batched-affine generation strategy.
fn worker_loop(prefixes: Vec<String>, suffixes: Vec<String>, infixes: Vec<String>, prefix_bounds: Vec<PrefixBound>, suffix_bounds: Vec<SuffixBound>, max_matches: i64, state: SharedState, table_arc: Arc<Vec<(FieldElement, FieldElement)>>, endo: Option<Endomorphism>, batch_size: usize) {

    // One read-only copy shared by every thread, hoisted to a plain slice
    // here so the hot loop below indexes it exactly as it would a
    // thread-local Vec — the Arc is never touched again after this line.
    // Sharing rather than cloning per thread matters for cache: the table
    // is ~38KB, and N private copies evict each other from the L2 that
    // sibling cores share, while one shared copy is read by all of them.
    let table: &[(FieldElement, FieldElement)] = table_arc.as_slice();

    // How many x-coordinates each computed point yields: 3 (x, beta*x,
    // beta^2*x) when the endomorphism constants validated at startup,
    // otherwise 1. Each is used with BOTH y parities, so the address
    // count per point is twice this. See the endomorphism section above.
    let (x_variants, beta, lambda) = match endo {
        Some(e) => (3usize, e.beta, e.lambda),
        None => (1usize, FieldElement::ONE, Scalar::ONE),
    };
    let lambda2 = lambda * lambda;
    let addrs_per_point = (x_variants * 2) as u64;

    let mut rng_source = thread_rng();
    let mut rng = ChaCha20Rng::from_rng(&mut rng_source).expect("failed to seed RNG");

    // Advancing the base by BATCH_SIZE each round keeps `base_scalar` and
    // `base_proj` in lockstep: base_proj is always base_scalar * G. Every
    // reported match re-derives its address from the scalar and checks it
    // against the matched address before printing, so any drift between
    // the two would be caught rather than producing a bad key.
    let step = ProjectivePoint::GENERATOR * Scalar::from(batch_size as u64);
    let mut base_scalar = Scalar::random(&mut rng);
    let mut base_proj = ProjectivePoint::GENERATOR * base_scalar;
    let mut offset: u64 = 0;

    // Reused across every batch — no per-batch allocation. This is the
    // only full-size buffer the batch needs; see
    // `batch_invert_prefix_products` for why the second one is gone.
    // Heap rather than a stack array so the size is a runtime choice —
    // allocated once per thread, never per batch.
    let mut products = vec![FieldElement::ONE; batch_size];
    // Both reused for every candidate — no per-candidate allocation on
    // the hot path.
    let mut addr_buf = [0u8; 34];
    let mut compressed = [0u8; 33];

    // Candidates are staged here until a full lane group is ready, so
    // their RIPEMD-160 hashes can be computed side by side. `staged_meta`
    // remembers which (batch index, x-variant, negated) each lane came
    // from, which is all that is needed to rebuild its private key if it
    // turns out to be a match. A batch always divides evenly into lane
    // groups — enforced at compile time next to `RIPEMD_LANES` — so this
    // never has a partial group left over at the end of one.
    let mut staged_sha = [[0u8; 32]; RIPEMD_LANES];
    let mut staged_meta = [(0u32, 0u8, false); RIPEMD_LANES];
    let mut staged = 0usize;
    // Padding written once here, never again — see `init_sha_block`.
    let mut sha_block = init_sha_block();

    'outer: loop {
        if max_matches != -1 && state.found_count.load(Ordering::Relaxed) >= max_matches {
            break;
        }
        // Checked once per batch, so it costs nothing measurable.
        if state.stop.load(Ordering::Relaxed) {
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

        // Forward pass. Fails only if some table point shares an
        // x-coordinate with the base (probability ~2^-256); rebase rather
        // than risk anything.
        let total_inv = match batch_invert_prefix_products(table, &px, &mut products) {
            Some(inv) => inv,
            None => {
                base_scalar = Scalar::random(&mut rng);
                base_proj = ProjectivePoint::GENERATOR * base_scalar;
                offset = 0;
                continue;
            }
        };

        // Backward pass, FUSED with the candidate work below rather than
        // run as a separate loop writing inverses back into a buffer.
        // Each inverse is consumed the instant it is produced, so it stays
        // in a register and never makes a round trip through memory, and
        // the table entry it needs is already loaded. Descending order is
        // what the trick requires; nothing downstream depends on the order
        // candidates are examined in.
        let mut running_inv = total_inv;
        for i in (0..batch_size).rev() {
            let (tx, ty) = table[i];

            // Recovering inv(d_i) from the running total: dividing out
            // everything above i leaves the product below it, so
            // multiplying by products[i-1] cancels down to exactly
            // inv(d_i). `d` is recomputed rather than stored — see
            // `batch_invert_prefix_products` for why.
            let d = tx - px;
            let inv_i = if i == 0 { running_inv } else { running_inv * products[i - 1] };
            running_inv = running_inv * d;

            // Standard affine addition: slope = (y2-y1)/(x2-x1),
            // x3 = slope² - x1 - x2, y3 = slope(x1-x3) - y1.
            //
            // No normalize_weak on slope or y_new: mul and square both
            // return magnitude-1 results, and the intermediate
            // subtractions stay within the budget mul accepts. x_new DOES
            // need one, because subtraction in k256 requires its
            // right-hand side to be magnitude-1 and x_new is used that
            // way just below. All of this is verified by running the
            // debug build, where k256's internal magnitude assertions are
            // active — they caught exactly this case when it was missing.
            let slope = (ty - py) * inv_i;
            let x_new = (slope.square() - px - tx).normalize_weak();
            let y_new = slope * (px - x_new) - py;

            // The y coordinate is needed only for its parity, and only
            // once: every one of the six addresses derived from this
            // point uses either this parity or its exact opposite. See
            // the endomorphism section above.
            let y_is_odd = bool::from(y_new.normalize().is_odd());

            let mut xv = x_new;
            for vx in 0..x_variants {
                // vx = 0 is x itself; each further step applies the
                // endomorphism again, one multiplication apiece.
                if vx > 0 {
                    xv = xv * beta;
                }
                compressed[1..33].copy_from_slice(&xv.normalize().to_bytes());

                // `neg` selects between the point and its negation, which
                // share these exact x bytes and differ only in parity.
                for neg in [false, true] {
                    compressed[0] = if y_is_odd != neg { 0x03 } else { 0x02 };

                    // SHA-256 runs one candidate at a time: the ARMv8
                    // crypto unit is a single pipe already saturated by
                    // one stream, so hashing two side by side measured no
                    // faster (21.24 vs 21.36 ns/hash) — unlike RIPEMD-160
                    // below, which has no hardware support and is short of
                    // parallelism rather than of throughput. Candidates
                    // are staged here so RIPEMD can take a whole group.
                    sha256_of_33(&mut sha_block, &compressed, &mut staged_sha[staged]);
                    staged_meta[staged] = (i as u32, vx as u8, neg);
                    staged += 1;
                    if staged < RIPEMD_LANES {
                        continue;
                    }
                    staged = 0;

                    let hashes = ripemd160_multi_32(&staged_sha);
                    for lane in 0..RIPEMD_LANES {
                        let mut first21 = [0u8; 21];
                        first21[0] = ADDRESS_VERSION_BYTE;
                        first21[1..21].copy_from_slice(&hashes[lane]);

                        // Fast path: a match requires every present
                        // category (prefix AND suffix AND infix) to be
                        // satisfied, so proving just ONE of them
                        // impossible is already enough to skip the
                        // checksum computation and Base58 encode entirely
                        // — regardless of what the others would say. Each
                        // bound set is empty (and so never triggers)
                        // whenever that category has no patterns, or
                        // lacks a usable bound (infix always lacks one;
                        // suffixes under 6 characters mathematically
                        // can't have one) — candidates then fall through
                        // to the exact path below for that category, same
                        // as before either optimization existed.
                        // Ordering and short-circuiting both matter here.
                        // These are written as a single `||` expression
                        // rather than two precomputed booleans so that a
                        // rejection by the first check skips the second
                        // entirely — computing both unconditionally
                        // measured ~20% slower whenever a prefix and a
                        // suffix were used together, because the suffix
                        // test performs a 21-byte modular reduction and
                        // was running even on candidates the prefix test
                        // had already rejected. The prefix test goes
                        // first since it is the cheaper of the two.
                        if (!prefix_bounds.is_empty()
                            && prefix_bounds.iter().all(|b| provably_outside_prefix(&first21, b)))
                            || (!suffix_bounds.is_empty()
                                && suffix_bounds.iter().all(|b| provably_outside_suffix(&first21, b)))
                        {
                            continue;
                        }

                        let addr = address_from_first21_into(&first21, &mut addr_buf);

                        let prefix_hit = match_any_prefix(&prefixes, &addr);
                        let suffix_hit = match_any_suffix(&suffixes, &addr);
                        let infix_hit = match_any_infix(&infixes, &addr);

                        if let (Some(prefix_hit), Some(suffix_hit), Some(infix_hit)) = (prefix_hit, suffix_hit, infix_hit) {
                            // Only reconstruct the actual private key on
                            // a real match (rare) — the hot loop above
                            // never needs it, since it only ever walks
                            // the public key forward.
                            //
                            // The lane's staged metadata says which
                            // candidate this was: `m_i` gives the batch
                            // index (table[i] is (i+1)*G, and base_scalar
                            // already advances by BATCH_SIZE each round),
                            // applying the endomorphism `m_vx` times
                            // multiplies the key by lambda^m_vx, and
                            // `m_neg` negates it. The self-check
                            // immediately below re-derives the address
                            // from whatever comes out of this and refuses
                            // to report on any disagreement, so an error
                            // here can only ever suppress a match, never
                            // emit a bad key.
                            let (m_i, m_vx, m_neg) = staged_meta[lane];
                            let base_key = base_scalar + Scalar::from((m_i + 1) as u64);
                            let mut sk_scalar = match m_vx {
                                1 => base_key * lambda,
                                2 => base_key * lambda2,
                                _ => base_key,
                            };
                            if m_neg {
                                sk_scalar = -sk_scalar;
                            }
                            let sk_bytes: [u8; 32] = sk_scalar.to_bytes().into();

                            // SELF-CHECK (defense in depth):
                            // independently re-derive the address
                            // straight from the reconstructed private key
                            // using k256's own scalar multiplication — a
                            // completely different code path from the
                            // batched affine arithmetic and multi-buffer
                            // hashing that produced this candidate. If
                            // they disagree, the key would not control
                            // the address, so refuse to report it rather
                            // than hand out something unusable. This
                            // costs one scalar multiplication, but only
                            // ever on a real match.
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

                            // Atomically claim a slot. Checking the count
                            // and then incrementing it as two separate
                            // steps let two threads both pass the check
                            // and both report, overshooting -m N.
                            // Claiming first and validating the claimed
                            // slot number makes that impossible: only
                            // claims below the limit are reported, and an
                            // over-limit claim is handed straight back.
                            let claimed = state.found_count.fetch_add(1, Ordering::Relaxed);
                            if max_matches != -1 && claimed >= max_matches {
                                state.found_count.fetch_sub(1, Ordering::Relaxed);
                                break 'outer;
                            }
                            let found_num = claimed + 1;
                            // Reset the progress-percentage baseline to
                            // start counting fresh from this match
                            // forward. fetch_max (rather than a plain
                            // store) keeps this correct even if two
                            // threads find matches close together and
                            // race here.
                            state
                                .last_match_tries
                                .fetch_max(state.keys_tried.load(Ordering::Relaxed), Ordering::Relaxed);

                            report_match(found_num, &desc, &addr, &wif, &priv_hex, &state);
                        }
                    }
                }
            }
        }

        // The compile-time check next to `RIPEMD_LANES` guarantees a
        // batch always ends on a lane-group boundary, so nothing is ever
        // left un-examined here. Re-checked in debug builds.
        debug_assert_eq!(staged, 0, "candidates left unexamined at end of batch");

        // Advance both representations of the base together, so
        // base_proj stays equal to base_scalar * G.
        base_scalar += Scalar::from(batch_size as u64);
        base_proj += step;
        offset += batch_size as u64;
        // Counts ADDRESSES examined, not base points walked — each point
        // yields `addrs_per_point` of them via the curve symmetries.
        state
            .keys_tried
            .fetch_add(batch_size as u64 * addrs_per_point, Ordering::Relaxed);

        if offset >= REBASE_INTERVAL {
            base_scalar = Scalar::random(&mut rng);
            base_proj = ProjectivePoint::GENERATOR * base_scalar;
            offset = 0;
        }
    }
}

// ===================== Stage benchmark =====================

/// Measures each stage of the pipeline on one thread and prints ns per
/// address, so tuning decisions rest on measurement instead of a model.
///
/// The decisive number is the RIPEMD-160 speedup: the multi-buffer
/// implementation is compared against the `ripemd` crate hashing the same
/// data one at a time. If that ratio is near `RIPEMD_LANES`, the lane
/// arrays compiled to real vector instructions; if it is near 1, they did
/// not and the target needs a different `RIPEMD_LANES` (or explicit
/// intrinsics). Nothing else in this program reveals that difference.
fn run_bench() {
    use rand::RngCore;
    use std::hint::black_box;

    println!("--- verus-vanity stage benchmark ---");
    println!("Single-threaded. Target: {}", std::env::consts::ARCH);
    println!("RIPEMD-160 lanes: {}\n", RIPEMD_LANES);
    println!("Note: the kernel is latency-bound, so the speedup below is");
    println!("expected to sit well under the lane count. What matters is");
    println!("whether ns/hash FALLS when lanes rise — that is chains");
    println!("filling each other's latency. It stops falling once the");
    println!("kernel becomes throughput-bound; that is the right width.\n");

    let mut rng = ChaCha20Rng::from_rng(&mut thread_rng()).expect("failed to seed RNG");

    // ---------- RIPEMD-160 ----------
    let mut inputs = [[0u8; 32]; RIPEMD_LANES];
    for slot in inputs.iter_mut() {
        rng.fill_bytes(slot);
    }

    let groups = 200_000usize;
    let hashes = groups * RIPEMD_LANES;

    let t = Instant::now();
    for _ in 0..groups {
        black_box(ripemd160_multi_32(black_box(&inputs)));
    }
    let multi_ns = t.elapsed().as_secs_f64() * 1e9 / hashes as f64;

    let t = Instant::now();
    for _ in 0..hashes {
        black_box(Ripemd160::digest(black_box(inputs[0])));
    }
    let single_ns = t.elapsed().as_secs_f64() * 1e9 / hashes as f64;

    println!("RIPEMD-160");
    println!("  one at a time (ripemd crate):  {:>8.2} ns/hash", single_ns);
    println!("  multi-buffer, {:>2} lanes:        {:>8.2} ns/hash", RIPEMD_LANES, multi_ns);
    println!("  speedup:                       {:>8.2}x", single_ns / multi_ns);
    println!();

    // ---------- SHA-256 ----------
    let mut pubkey = [0u8; 33];
    rng.fill_bytes(&mut pubkey);
    let mut sha_block = init_sha_block();
    let mut digest_out = [0u8; 32];

    let t = Instant::now();
    for _ in 0..hashes {
        sha256_of_33(&mut sha_block, black_box(&pubkey), &mut digest_out);
        black_box(&digest_out);
    }
    let sha_direct_ns = t.elapsed().as_secs_f64() * 1e9 / hashes as f64;

    let t = Instant::now();
    for _ in 0..hashes {
        black_box(Sha256::digest(black_box(pubkey)));
    }
    let sha_crate_ns = t.elapsed().as_secs_f64() * 1e9 / hashes as f64;

    println!("SHA-256 (33-byte message, one block)");
    println!("  via Sha256::digest:            {:>8.2} ns/hash", sha_crate_ns);
    println!("  direct compress256:            {:>8.2} ns/hash", sha_direct_ns);
    println!("  saved:                         {:>8.2} ns/hash", sha_crate_ns - sha_direct_ns);
    println!();

    // ---------- Elliptic curve + serialization ----------
    let endo = setup_endomorphism();
    let (x_variants, beta) = match endo {
        Some(e) => (3usize, e.beta),
        None => (1usize, FieldElement::ONE),
    };
    let table = build_addition_table(BATCH_SIZE);
    let mut products = vec![FieldElement::ONE; BATCH_SIZE];
    let base_scalar = Scalar::random(&mut rng);
    let (px, py) = affine_xy(&(ProjectivePoint::GENERATOR * base_scalar).to_affine())
        .expect("base point is valid");

    let batches = 300usize;
    let ec_addresses = batches * BATCH_SIZE * x_variants * 2;
    let mut compressed = [0u8; 33];

    let t = Instant::now();
    for _ in 0..batches {
        let total_inv = batch_invert_prefix_products(&table, &px, &mut products)
            .expect("no zero denominators");
        let mut running_inv = total_inv;
        for i in (0..BATCH_SIZE).rev() {
            let (tx, ty) = table[i];
            let d = tx - px;
            let inv_i = if i == 0 { running_inv } else { running_inv * products[i - 1] };
            running_inv = running_inv * d;

            let slope = (ty - py) * inv_i;
            let x_new = (slope.square() - px - tx).normalize_weak();
            let y_new = slope * (px - x_new) - py;
            let y_is_odd = bool::from(y_new.normalize().is_odd());

            let mut xv = x_new;
            for vx in 0..x_variants {
                if vx > 0 {
                    xv = xv * beta;
                }
                compressed[1..33].copy_from_slice(&xv.normalize().to_bytes());
                for neg in [false, true] {
                    compressed[0] = if y_is_odd != neg { 0x03 } else { 0x02 };
                    black_box(&compressed);
                }
            }
        }
    }
    let ec_ns = t.elapsed().as_secs_f64() * 1e9 / ec_addresses as f64;

    // Serialization alone, to split the stage above into field arithmetic
    // versus turning an x-coordinate into 32 bytes. One normalize +
    // to_bytes serves both parities, so it is charged to two addresses.
    let ser_iters = 2_000_000usize;
    let t = Instant::now();
    for _ in 0..ser_iters {
        compressed[1..33].copy_from_slice(&black_box(px).normalize().to_bytes());
        black_box(&compressed);
    }
    let ser_ns = t.elapsed().as_secs_f64() * 1e9 / (ser_iters * 2) as f64;

    println!("Elliptic curve + public-key serialization");
    println!("  per address ({} per point):     {:>8.2} ns", x_variants * 2, ec_ns);
    println!("    of which x -> 32 bytes:      {:>8.2} ns", ser_ns);
    println!("    field arithmetic:            {:>8.2} ns", ec_ns - ser_ns);
    println!();

    // ---------- Full candidate pipeline ----------
    // Everything the worker does per candidate short of reporting a
    // match: the stages above, plus lane staging, first21 assembly and
    // the prefix pre-filter. Measured rather than inferred, so the glue
    // between the stages stops being a number arrived at by subtracting
    // from the observed rate.
    let filter_bounds: Vec<PrefixBound> = compute_prefix_bound("RDarkt").into_iter().collect();
    let mut staged_sha = [[0u8; 32]; RIPEMD_LANES];
    let mut staged = 0usize;
    let mut survivors = 0usize;

    let t = Instant::now();
    for _ in 0..batches {
        let total_inv = batch_invert_prefix_products(&table, &px, &mut products)
            .expect("no zero denominators");
        let mut running_inv = total_inv;
        for i in (0..BATCH_SIZE).rev() {
            let (tx, ty) = table[i];
            let d = tx - px;
            let inv_i = if i == 0 { running_inv } else { running_inv * products[i - 1] };
            running_inv = running_inv * d;

            let slope = (ty - py) * inv_i;
            let x_new = (slope.square() - px - tx).normalize_weak();
            let y_new = slope * (px - x_new) - py;
            let y_is_odd = bool::from(y_new.normalize().is_odd());

            let mut xv = x_new;
            for vx in 0..x_variants {
                if vx > 0 {
                    xv = xv * beta;
                }
                compressed[1..33].copy_from_slice(&xv.normalize().to_bytes());
                for neg in [false, true] {
                    compressed[0] = if y_is_odd != neg { 0x03 } else { 0x02 };

                    sha256_of_33(&mut sha_block, &compressed, &mut staged_sha[staged]);
                    staged += 1;
                    if staged < RIPEMD_LANES {
                        continue;
                    }
                    staged = 0;

                    let group = ripemd160_multi_32(&staged_sha);
                    for digest in group.iter() {
                        let mut first21 = [0u8; 21];
                        first21[0] = ADDRESS_VERSION_BYTE;
                        first21[1..21].copy_from_slice(digest);

                        if !filter_bounds.is_empty()
                            && filter_bounds.iter().all(|b| provably_outside_prefix(&first21, b))
                        {
                            continue;
                        }
                        survivors += 1;
                        black_box(&first21);
                    }
                }
            }
        }
    }
    let full_ns = t.elapsed().as_secs_f64() * 1e9 / ec_addresses as f64;
    black_box(survivors);

    let stages = ec_ns + sha_direct_ns + multi_ns;
    let overhead = full_ns - stages;

    println!("Full candidate pipeline");
    println!("  per address:                   {:>8.2} ns", full_ns);
    println!("  stages above account for:      {:>8.2} ns", stages);
    println!("  staging + first21 + filter:    {:>8.2} ns  ({:>4.1}%)", overhead, 100.0 * overhead / full_ns);
    println!();

    // ---------- Totals ----------
    let threads = num_cpus::get();
    println!("Where the time goes, per address");
    println!("  EC + serialize:                {:>8.2} ns  ({:>4.1}%)", ec_ns, 100.0 * ec_ns / full_ns);
    println!("  SHA-256:                       {:>8.2} ns  ({:>4.1}%)", sha_direct_ns, 100.0 * sha_direct_ns / full_ns);
    println!("  RIPEMD-160:                    {:>8.2} ns  ({:>4.1}%)", multi_ns, 100.0 * multi_ns / full_ns);
    println!("  glue (staging/first21/filter): {:>8.2} ns  ({:>4.1}%)", overhead, 100.0 * overhead / full_ns);
    println!("  ---------------------------------------");
    println!("  total:                         {:>8.2} ns/address", full_ns);
    println!("  implies ~{:.1} MW/s on 1 core, in this loop", 1000.0 / full_ns);
    println!();

    // ---------- Measured, not extrapolated ----------
    // An earlier version printed the line above multiplied by the core
    // count and called it the machine's throughput. That was wrong twice
    // over, and the two errors compounded into roughly a factor of two:
    //
    //   * it assumed perfect scaling, when per-thread throughput actually
    //     falls as cores are added (shared cache, memory bandwidth, and on
    //     a phone a package power budget that lowers clocks under load);
    //   * a tight microbenchmark loop keeps its working set hot in a way
    //     the real search does not, so even the single-core figure is
    //     optimistic.
    //
    // So rather than model any of that, just run the real search and
    // report what it does.
    println!("Measured on the real search (not extrapolated)");
    let dur = Duration::from_millis(600);
    let one = median_of(3, || measure_throughput(BATCH_SIZE, 1, dur, None));
    println!("  1 thread:                      {:>8.2} MW/s", one / 1e6);
    if threads > 1 {
        let many = median_of(3, || measure_throughput(BATCH_SIZE, threads, dur, None));
        println!("  {} threads:                     {:>8.2} MW/s", threads, many / 1e6);
        let ideal = one * threads as f64;
        println!(
            "  scaling:                       {:>8.0}% of linear ({:.2} MW/s per thread)",
            100.0 * many / ideal,
            many / threads as f64 / 1e6
        );
        println!();
        println!("The gap between this and the per-address figure above is");
        println!("real work the microbenchmark does not see, plus whatever");
        println!("the scaling percentage is short of 100. Try -b and -t to");
        println!("find what this machine likes; -b in the low thousands is a");
        println!("reasonable place to start.");
    }
}

// ===================== Clustering =====================
//
// Several devices searching for the same patterns, with one of them
// holding the objective and collecting the results.
//
// This problem needs far less coordination than distributed work usually
// does, and the design leans on that. Every thread already begins at a
// random 256-bit scalar and walks forward from there, so two machines
// covering the same ground would require them to land within a batch of
// each other in a space of 2^256 — it does not happen. That removes the
// work queue, the range assignment, the duplicate detection and the
// synchronisation that a cluster would normally need. A worker only has
// to learn what to look for and have somewhere to send a hit.
//
// So the protocol is small and line-based over plain TCP, with no
// dependencies beyond std: patterns are Base58, which cannot contain a
// space or a comma, so framing needs nothing clever.
//
//   worker -> master   JOIN <proto> <pass> <name>
//   master -> worker   WORK <max_matches> <prefixes> <infixes> <suffixes>
//                      DENY <reason>
//   worker -> master   RATE <addresses-since-last-report>
//                      MATCH <address> <wif> <hex> <desc>
//   master -> worker   STOP        stop searching (a keepalive worker idles
//                                  and waits for the next objective)
//                      SHUTDOWN    exit, whatever --keepalive says
//
// `-` stands for an empty pattern list, and a description has its spaces
// replaced by '_' so a record stays on one line.
//
// EVERY MATCH A MASTER RECEIVES IS RE-DERIVED AND RE-CHECKED before it
// counts (see `verify_remote_match`). A worker is a different machine
// running a binary the master cannot vouch for, so its claim is treated
// as a claim: the master recomputes the address from the private key it
// was sent and confirms both that they agree and that the result
// actually satisfies the requested patterns. A corrupted transfer, a
// mismatched build or a tampered record all fail that check, and no
// forged record can pass it without having done the work.
//
// ON PRIVACY: a match carries a private key, and this protocol sends it
// in the clear. Anyone able to watch the traffic can take the wallet.
// That is acceptable on a network you control and not otherwise, so both
// ends say so at startup. `--keys-stay-local` keeps keys on the machine
// that found them and sends only the address, at the cost of having to
// collect the file from that device afterwards. For an untrusted path,
// forward a port over SSH rather than trusting this.

const CLUSTER_PROTO: &str = "v1";

/// Longest protocol line accepted from a peer. Every legitimate message
/// is well under this; the cap exists because `read_line` will otherwise
/// grow its buffer for as long as a peer keeps sending without a
/// newline, which is a way to exhaust memory from the far end of a
/// socket.
const MAX_LINE_BYTES: u64 = 8 * 1024;

/// `read_line` with a ceiling. Returns Ok(0) at end of stream, and an
/// error if the peer exceeds the cap so the caller drops the connection.
fn read_line_capped(reader: &mut BufReader<TcpStream>, buf: &mut String) -> std::io::Result<usize> {
    buf.clear();
    let n = std::io::Read::take(reader.by_ref(), MAX_LINE_BYTES).read_line(buf)?;
    if n as u64 >= MAX_LINE_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "peer sent an over-long line",
        ));
    }
    Ok(n)
}

/// A found keypair, as passed between threads and over the wire.
#[derive(Clone)]
struct MatchRecord {
    desc: String,
    addr: String,
    wif: String,
    priv_hex: String,
}

fn encode_list(items: &[String]) -> String {
    if items.is_empty() {
        "-".to_string()
    } else {
        items.join(",")
    }
}

fn decode_list(field: &str) -> Vec<String> {
    if field == "-" {
        Vec::new()
    } else {
        field.split(',').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
    }
}

/// Re-derives the address from a received private key and confirms it
/// matches both the claim and the search patterns. See the module note.
/// True when the worker deliberately withheld the key (`--keys-stay-local`).
fn is_held_remotely(rec: &MatchRecord) -> bool {
    rec.wif == "-" && rec.priv_hex == "-"
}

/// Checks that a string really is a well-formed VerusCoin address:
/// Base58Check of 25 bytes, right version, intact checksum. Used for
/// matches whose key stayed on the worker, where there is no key to
/// derive from.
fn validate_address_string(addr: &str) -> Result<(), String> {
    let raw = bs58::decode(addr).into_vec().map_err(|_| "not valid Base58".to_string())?;
    if raw.len() != 25 {
        return Err(format!("decodes to {} bytes, expected 25", raw.len()));
    }
    if raw[0] != ADDRESS_VERSION_BYTE {
        return Err(format!("version byte is 0x{:02x}, expected 0x{:02x}", raw[0], ADDRESS_VERSION_BYTE));
    }
    let checksum = Sha256::digest(Sha256::digest(&raw[0..21]));
    if checksum[0..4] != raw[21..25] {
        return Err("checksum does not match".into());
    }
    Ok(())
}

fn verify_remote_match(
    rec: &MatchRecord,
    prefixes: &[String],
    infixes: &[String],
    suffixes: &[String],
) -> Result<(), String> {
    // A worker running --keys-stay-local sends no key on purpose, so
    // there is nothing to derive from. Verify what can be verified: that
    // the address is genuinely well-formed and is what was asked for.
    //
    // This is deliberately weaker — the master cannot prove the worker
    // actually holds the key, only that it is quoting a real address that
    // fits. That is the trade being made by keeping keys off the network,
    // and the report says so.
    if is_held_remotely(rec) {
        validate_address_string(&rec.addr)?;
        let p_ok = match_any_prefix(prefixes, &rec.addr).is_some();
        let i_ok = match_any_infix(infixes, &rec.addr).is_some();
        let s_ok = match_any_suffix(suffixes, &rec.addr).is_some();
        if !(p_ok && i_ok && s_ok) {
            return Err(format!("{} does not satisfy the requested patterns", rec.addr));
        }
        return Ok(());
    }

    let bytes = match hex::decode(&rec.priv_hex) {
        Ok(b) => b,
        Err(_) => return Err("private key is not valid hex".into()),
    };
    if bytes.len() != 32 {
        return Err(format!("private key is {} bytes, expected 32", bytes.len()));
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);

    let scalar = match Option::<Scalar>::from(Scalar::from_repr(arr.into())) {
        Some(sc) => sc,
        None => return Err("private key is not a valid scalar".into()),
    };
    let point = ProjectivePoint::GENERATOR * scalar;
    let compressed = point.to_affine().to_encoded_point(true);
    let derived = address_from_first21(&compute_first21(compressed.as_bytes(), ADDRESS_VERSION_BYTE));

    if derived != rec.addr {
        return Err(format!("key derives to {}, not the claimed {}", derived, rec.addr));
    }
    // The address being real is not enough — it also has to be what was
    // asked for, or a worker running stale patterns would pollute the
    // results.
    let p_ok = match_any_prefix(prefixes, &derived).is_some();
    let i_ok = match_any_infix(infixes, &derived).is_some();
    let s_ok = match_any_suffix(suffixes, &derived).is_some();
    if !(p_ok && i_ok && s_ok) {
        return Err(format!("{} does not satisfy the requested patterns", derived));
    }
    if private_key_to_wif(&arr, WIF_VERSION_BYTE, true) != rec.wif {
        return Err("WIF does not match the private key".into());
    }
    Ok(())
}

/// Rejects a pass phrase that cannot survive the wire format.
///
/// JOIN is a space-delimited line, so a pass containing whitespace is
/// split across fields: both ends can give the identical --pass and
/// still fail with "bad-pass", while the worker name is mangled too.
/// Catching it at startup turns a confusing authentication failure into
/// a clear message.
fn validate_token_or_exit(token: &str) {
    if token.chars().any(|c| c.is_whitespace()) {
        eprintln!("⚠️  --pass cannot contain spaces or tabs.");
        eprintln!("   The join message is space-delimited, so a token with whitespace");
        eprintln!("   is split apart and never matches — even when both sides pass the");
        eprintln!("   same thing. Use a single word, e.g. --pass my-secret");
        std::process::exit(1);
    }
}

fn write_line(stream: &mut TcpStream, line: &str) -> std::io::Result<()> {
    stream.write_all(line.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()
}

// ---------- master ----------

/// Serves the objective to workers and collects their results. Returns
/// once enough matches exist; the local search runs alongside.
fn spawn_cluster_master(
    listen: String,
    token: String,
    config_prefixes: Vec<String>,
    config_infixes: Vec<String>,
    config_suffixes: Vec<String>,
    max_matches: i64,
    state: SharedState,
    stop_workers: bool,
) -> Arc<Mutex<Vec<TcpStream>>> {
    // What a finished objective means for the workers. STOP lets a
    // `--keepalive` worker idle and wait for the next job; SHUTDOWN ends
    // it regardless, which is how a cluster gets dismissed on purpose.
    let final_msg: &str = if stop_workers { "SHUTDOWN" } else { "STOP" };

    // Live worker connections, so the master can notify them before it
    // exits. Relying on the per-worker handler to do it loses the race:
    // the moment the objective is met the search threads finish, main
    // returns and the process dies, taking the handler with it before it
    // has written anything. The worker then sees a bare disconnect and,
    // if it is keepalive, goes back to waiting — which is exactly what
    // --stop-workers is meant to prevent.
    let conns: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));
    let conns_for_listener = Arc::clone(&conns);
    let listener = match TcpListener::bind(&listen) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("⚠️  Could not listen on {}: {}", listen, e);
            std::process::exit(1);
        }
    };
    println!("Cluster master listening on {}", listen);
    if token.is_empty() {
        println!("  no --pass set: any device that can reach this port may join");
    }
    println!(
        "  workers send private keys in the clear — use this on a network you\n  \
         control, or forward the port over SSH"
    );
    println!();

    thread::spawn(move || {
        for incoming in listener.incoming() {
            let mut stream = match incoming {
                Ok(s) => s,
                Err(_) => continue,
            };
            let peer = stream
                .peer_addr()
                .map(|a| a.to_string())
                .unwrap_or_else(|_| "unknown".into());

            let token = token.clone();
            let final_msg = final_msg.to_string();
            let conns = Arc::clone(&conns_for_listener);
            let p = config_prefixes.clone();
            let i = config_infixes.clone();
            let sfx = config_suffixes.clone();
            let st = state.clone();

            thread::spawn(move || {
                let reader_stream = match stream.try_clone() {
                    Ok(s) => s,
                    Err(_) => return,
                };
                let mut reader = BufReader::new(reader_stream);
                let mut first = String::new();
                if read_line_capped(&mut reader, &mut first).is_err() {
                    return;
                }

                let parts: Vec<&str> = first.trim().split(' ').collect();
                if parts.len() < 4 || parts[0] != "JOIN" {
                    let _ = write_line(&mut stream, "DENY malformed-join");
                    return;
                }
                if parts[1] != CLUSTER_PROTO {
                    let _ = write_line(&mut stream, "DENY protocol-mismatch");
                    eprintln!("Rejected {}: protocol {} (this build speaks {})", peer, parts[1], CLUSTER_PROTO);
                    return;
                }
                if parts[2] != token {
                    let _ = write_line(&mut stream, "DENY bad-pass");
                    eprintln!("Rejected {}: wrong pass", peer);
                    return;
                }
                let name = parts[3].to_string();

                let work = format!(
                    "WORK {} {} {} {}",
                    max_matches,
                    encode_list(&p),
                    encode_list(&i),
                    encode_list(&sfx)
                );
                if write_line(&mut stream, &work).is_err() {
                    return;
                }
                println!("Worker joined: {} ({})", name, peer);
                if let Ok(mut list) = conns.lock() {
                    if let Ok(c) = stream.try_clone() {
                        list.push(c);
                    }
                }

                let mut line = String::new();
                loop {
                    line.clear();
                    match read_line_capped(&mut reader, &mut line) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {}
                    }
                    let t = line.trim();
                    if let Some(rest) = t.strip_prefix("RATE ") {
                        if let Ok(n) = rest.trim().parse::<u64>() {
                            st.remote_tried.fetch_add(n, Ordering::Relaxed);
                            // Accumulate: with several workers reporting,
                            // storing would leave only whichever spoke
                            // last and undercount the cluster. The stats
                            // thread drains this each tick.
                            st.remote_rate.fetch_add(n, Ordering::Relaxed);
                        }
                    } else if let Some(rest) = t.strip_prefix("MATCH ") {
                        let f: Vec<&str> = rest.split(' ').collect();
                        if f.len() < 4 {
                            continue;
                        }
                        let rec = MatchRecord {
                            addr: f[0].to_string(),
                            wif: f[1].to_string(),
                            priv_hex: f[2].to_string(),
                            desc: f[3].replace('_', " "),
                        };
                        match verify_remote_match(&rec, &p, &i, &sfx) {
                            Ok(()) => {
                                let claimed = st.found_count.fetch_add(1, Ordering::Relaxed);
                                if max_matches != -1 && claimed >= max_matches {
                                    st.found_count.fetch_sub(1, Ordering::Relaxed);
                                    let _ = write_line(&mut stream, &final_msg);
                                    break;
                                }
                                st.last_match_tries
                                    .store(st.keys_tried.load(Ordering::Relaxed), Ordering::Relaxed);
                                report_match(
                                    claimed + 1,
                                    &format!("{} (from {})", rec.desc, name),
                                    &rec.addr,
                                    &rec.wif,
                                    &rec.priv_hex,
                                    &st,
                                );
                            }
                            Err(why) => {
                                eprintln!(
                                    "⚠️  Rejected a match from {}: {}\n   \
                                     It was NOT counted. A worker on a different build or a\n   \
                                     corrupted transfer would both look like this.",
                                    name, why
                                );
                            }
                        }
                    } else if t == "BYE" {
                        break;
                    }

                    if max_matches != -1 && st.found_count.load(Ordering::Relaxed) >= max_matches {
                        let _ = write_line(&mut stream, &final_msg);
                        break;
                    }
                }
                println!("Worker left: {}", name);
            });
        }
    });

    conns
}

// ---------- worker ----------

/// Serves SHUTDOWN to every worker that connects, so a set of keepalive
/// workers can be dismissed without running a search. Runs until
/// interrupted, because workers reconnect on their own schedule and there
/// is no way to know how many are out there.
fn run_dismiss(listen: &str, token: &str) {
    let listener = match TcpListener::bind(listen) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("⚠️  Could not listen on {}: {}", listen, e);
            std::process::exit(1);
        }
    };
    println!("Dismissing workers on {} — press Ctrl+C when they have all gone.", listen);
    println!("A keepalive worker retries every few seconds, so give it a moment.\n");

    for incoming in listener.incoming() {
        let mut stream = match incoming {
            Ok(s) => s,
            Err(_) => continue,
        };
        let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_else(|_| "unknown".into());
        let reader_stream = match stream.try_clone() {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut reader = BufReader::new(reader_stream);
        let mut first = String::new();
        if read_line_capped(&mut reader, &mut first).is_err() {
            continue;
        }
        let parts: Vec<&str> = first.trim().split(' ').collect();
        if parts.len() >= 4 && parts[0] == "JOIN" && parts[2] == token {
            let _ = write_line(&mut stream, "SHUTDOWN");
            println!("Dismissed {} ({})", parts[3], peer);
        } else {
            let _ = write_line(&mut stream, "DENY bad-pass");
            println!("Refused {} (wrong pass)", peer);
        }
    }
}

/// Why a round of work ended.
enum WorkerEnd {
    /// Master says the objective is met. A keepalive worker waits for the
    /// next one; otherwise this is the end.
    Stopped,
    /// Master says exit, regardless of keepalive.
    Shutdown,
    /// Connection lost. Treated like Stopped, since a master that was
    /// interrupted looks exactly like this from here.
    Disconnected,
}

/// Connects to a master, adopts its objective, and searches until told to
/// stop. Never prints or stores a match itself: the master owns results.
///
/// With `keepalive`, finishing a round does not end the process — the
/// worker goes back to waiting for a master. That is what makes a phone
/// usable as a standing member of a cluster instead of something to be
/// restarted by hand for every job.
fn run_cluster_worker(
    addr: &str,
    token: &str,
    name: &str,
    threads: usize,
    batch_size: usize,
    keys_stay_local: bool,
    keepalive: bool,
) {
    println!("--- verus-vanity cluster worker ---");
    if keepalive {
        println!("Keepalive: will wait for further objectives until told to shut down.");
    }

    // Built once and reused for every round; entry i is (i+1)*G regardless
    // of the objective, so a new job does not need a new table.
    let table = Arc::new(build_addition_table(batch_size));
    let endo = setup_endomorphism();
    let mut announced_wait = false;

    // How long to wait between connection attempts. A failed connect to a
    // closed port costs almost nothing, so this is kept short: the gap is
    // also the worst-case delay before a worker picks up a new objective,
    // and a longer one can miss a short job entirely.
    const RETRY_SECS: u64 = 2;

    loop {
        let stream = match TcpStream::connect(addr) {
            Ok(s) => s,
            Err(e) => {
                if !keepalive {
                    eprintln!("⚠️  Could not connect to {}: {}", addr, e);
                    std::process::exit(1);
                }
                if !announced_wait {
                    println!("No master at {} — retrying every {}s. ({})", addr, RETRY_SECS, e);
                    announced_wait = true;
                }
                thread::sleep(Duration::from_secs(RETRY_SECS));
                continue;
            }
        };
        announced_wait = false;

        match run_worker_round(stream, addr, token, name, threads, batch_size, keys_stay_local, &table, endo) {
            WorkerEnd::Shutdown => {
                println!("Master dismissed this worker. Exiting.");
                return;
            }
            WorkerEnd::Stopped | WorkerEnd::Disconnected if keepalive => {
                println!("Round over — waiting for the next objective.\n");
                // Brief pause so a master that is shutting down has
                // finished releasing the port before the next attempt.
                thread::sleep(Duration::from_millis(500));
                continue;
            }
            _ => return,
        }
    }
}

/// One round: join, take the objective, search, report, and return the
/// reason it ended.
#[allow(clippy::too_many_arguments)]
fn run_worker_round(
    mut stream: TcpStream,
    addr: &str,
    token: &str,
    name: &str,
    threads: usize,
    batch_size: usize,
    keys_stay_local: bool,
    table: &Arc<Vec<(FieldElement, FieldElement)>>,
    endo: Option<Endomorphism>,
) -> WorkerEnd {
    println!("Connected to master at {} as \"{}\".", addr, name);

    let reader_stream = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return WorkerEnd::Disconnected,
    };
    let mut reader = BufReader::new(reader_stream);

    if write_line(&mut stream, &format!("JOIN {} {} {}", CLUSTER_PROTO, token, name)).is_err() {
        return WorkerEnd::Disconnected;
    }

    let mut line = String::new();
    if read_line_capped(&mut reader, &mut line).is_err() {
        return WorkerEnd::Disconnected;
    }
    let t = line.trim().to_string();
    if t == "SHUTDOWN" {
        return WorkerEnd::Shutdown;
    }
    if let Some(reason) = t.strip_prefix("DENY ") {
        eprintln!("⚠️  Master refused this worker: {}", reason);
        if reason == "bad-pass" {
            eprintln!("   Pass the same --pass the master was started with.");
        }
        // A wrong token will not fix itself by retrying.
        std::process::exit(1);
    }
    let f: Vec<&str> = t.split(' ').collect();
    if f.len() < 5 || f[0] != "WORK" {
        eprintln!("⚠️  Unexpected reply from master: {}", t);
        return WorkerEnd::Disconnected;
    }
    let prefixes = decode_list(f[2]);
    let infixes = decode_list(f[3]);
    let suffixes = decode_list(f[4]);
    let max_matches: i64 = f[1].parse().unwrap_or(-1);

    println!("Objective:");
    if !prefixes.is_empty() {
        println!("  prefixes: {}", prefixes.join(", "));
    }
    if !infixes.is_empty() {
        println!("  infixes:  {}", infixes.join(", "));
    }
    if !suffixes.is_empty() {
        println!("  suffixes: {}", suffixes.join(", "));
    }
    println!(
        "  matches wanted: {}",
        if max_matches == -1 { "unlimited".to_string() } else { max_matches.to_string() }
    );
    if keys_stay_local {
        println!("  --keys-stay-local: only addresses are sent; keys stay in this device's output");
    } else {
        println!("  found keys are sent to the master in the clear");
    }
    println!("Threads: {}   Batch: {}", threads, batch_size);
    println!("-----------------------------");

    let (tx, rx) = std::sync::mpsc::channel::<MatchRecord>();
    let mut state = SharedState::new(&None);
    state.match_sink = Some(tx);

    let prefix_bounds = try_compute_all_prefix_bounds(&prefixes);
    let suffix_bounds = try_compute_all_suffix_bounds(&suffixes);

    let mut handles = Vec::new();
    for _ in 0..threads {
        let p = prefixes.clone();
        let i = infixes.clone();
        let sfx = suffixes.clone();
        let pb = prefix_bounds.clone();
        let sb = suffix_bounds.clone();
        let st = state.clone();
        let tb = Arc::clone(table);
        handles.push(thread::spawn(move || {
            worker_loop(p, sfx, i, pb, sb, -1, st, tb, endo, batch_size);
        }));
    }

    // Reader thread: records why the round ended.
    let stop_flag = Arc::clone(&state.stop);
    let shutdown_seen = Arc::new(AtomicBool::new(false));
    let disconnected = Arc::new(AtomicBool::new(false));
    {
        let sd = Arc::clone(&shutdown_seen);
        let dc = Arc::clone(&disconnected);
        thread::spawn(move || {
            let mut l = String::new();
            loop {
                l.clear();
                match read_line_capped(&mut reader, &mut l) {
                    Ok(0) | Err(_) => {
                        dc.store(true, Ordering::Relaxed);
                        stop_flag.store(true, Ordering::Relaxed);
                        return;
                    }
                    Ok(_) => {}
                }
                match l.trim() {
                    "STOP" => {
                        stop_flag.store(true, Ordering::Relaxed);
                        return;
                    }
                    "SHUTDOWN" => {
                        sd.store(true, Ordering::Relaxed);
                        stop_flag.store(true, Ordering::Relaxed);
                        return;
                    }
                    _ => {}
                }
            }
        });
    }

    let mut last_reported = 0u64;
    let start = Instant::now();
    let mut lost = false;
    loop {
        thread::sleep(Duration::from_secs(1));

        while let Ok(rec) = rx.try_recv() {
            let payload = if keys_stay_local {
                println!("MATCH kept on this device: {}", rec.addr);
                println!("  WIF: {}", rec.wif);
                println!("  Private Key (hex): {}", rec.priv_hex);
                format!("MATCH {} - - {}", rec.addr, rec.desc.replace(' ', "_"))
            } else {
                format!(
                    "MATCH {} {} {} {}",
                    rec.addr, rec.wif, rec.priv_hex, rec.desc.replace(' ', "_")
                )
            };
            let _ = write_line(&mut stream, &payload);
        }

        let total = state.keys_tried.load(Ordering::Relaxed);
        let delta = total.saturating_sub(last_reported);
        last_reported = total;
        if write_line(&mut stream, &format!("RATE {}", delta)).is_err() {
            lost = true;
        } else {
            println!(
                "Contributed: {} ({} total in {:.0}s)",
                format_with_si_rate(delta),
                format_with_si(total),
                start.elapsed().as_secs_f64()
            );
        }

        if lost || state.stop.load(Ordering::Relaxed) {
            break;
        }
    }

    // Let the search threads notice the flag and finish, so a keepalive
    // round does not leave them running behind the next one.
    state.stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }
    let _ = write_line(&mut stream, "BYE");

    if shutdown_seen.load(Ordering::Relaxed) {
        WorkerEnd::Shutdown
    } else if disconnected.load(Ordering::Relaxed) || lost {
        println!("Master went away.");
        WorkerEnd::Disconnected
    } else {
        println!("Master says the objective is met.");
        WorkerEnd::Stopped
    }
}

// ===================== Entry point =====================

fn main() {
    let config = parse_cli();
    let expected = expected_tries(&config.prefixes, &config.suffixes, &config.infixes);

    // Proven correct before it is trusted; None simply means the search
    // runs on negation symmetry alone. See `setup_endomorphism`.
    let endo = setup_endomorphism();

    print_banner(&config, endo.is_some());

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

    // Precompute G, 2G, ..., BATCH_SIZE*G once. Every thread shares this
    // single read-only copy (~40KB): each worker hoists it to a plain
    // slice before its hot loop, so there is no shared-pointer
    // indirection in the loop itself, and one copy stays resident in the
    // caches that sibling cores share instead of N copies evicting one
    // another.
    let table = Arc::new(build_addition_table(config.batch_size));

    let state = SharedState::new(&config.output_file);
    let start_time = Instant::now();
    let target_matches: f64 = if config.max_matches == -1 { 1.0 } else { config.max_matches.max(1) as f64 };

    // Master mode: serve the objective and collect verified results while
    // this machine searches alongside the workers.
    let mut worker_conns: Option<Arc<Mutex<Vec<TcpStream>>>> = None;
    if let Some(listen) = config.serve.clone() {
        worker_conns = Some(spawn_cluster_master(
            listen,
            config.token.clone(),
            config.prefixes.clone(),
            config.infixes.clone(),
            config.suffixes.clone(),
            config.max_matches,
            state.clone(),
            config.stop_workers,
        ));
    }

    spawn_stats_thread(state.clone(), start_time, expected, target_matches, config.eta_every);

    let mut handles = Vec::new();
    for _ in 0..config.threads {
        let prefixes = config.prefixes.clone();
        let suffixes = config.suffixes.clone();
        let infixes = config.infixes.clone();
        let prefix_bounds = prefix_bounds.clone();
        let suffix_bounds = suffix_bounds.clone();
        let state = state.clone();
        let max_matches = config.max_matches;
        let batch_size = config.batch_size;
        // Fastest core first, cycling if there are more threads than
        // detected cores (e.g. -t set above the core count).
        let table = Arc::clone(&table);
        handles.push(thread::spawn(move || {
            worker_loop(prefixes, suffixes, infixes, prefix_bounds, suffix_bounds, max_matches, state, table, endo, batch_size)
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Objective met: tell the workers directly rather than letting them
    // infer it from the socket closing. STOP lets a keepalive worker idle
    // for the next job; SHUTDOWN ends it. Without this the process would
    // exit first and they would only ever see a disconnect.
    if let Some(conns) = worker_conns {
        let msg = if config.stop_workers { "SHUTDOWN" } else { "STOP" };
        if let Ok(mut list) = conns.lock() {
            for c in list.iter_mut() {
                let _ = write_line(c, msg);
            }
            if !list.is_empty() {
                println!(
                    "Told {} worker(s) to {}.",
                    list.len(),
                    if config.stop_workers { "shut down" } else { "stand by" }
                );
            }
        }
        // Give the writes a moment to land before the process goes away.
        thread::sleep(Duration::from_millis(300));
    }
}

// ===================== Tests =====================
//
// These exist to check the optimizations that are easy to get subtly
// wrong and hard to notice: each one verifies a fast path against an
// independent, obviously-correct implementation rather than against
// itself. Run with `cargo test`.

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// Derives an address the slow, obvious way: private key → k256's own
    /// scalar multiplication → compressed public key → Base58Check. This
    /// is the reference every fast path is checked against.
    fn address_for_scalar(s: &Scalar) -> String {
        let point = (ProjectivePoint::GENERATOR * *s).to_affine();
        let compressed = point.to_encoded_point(true);
        address_from_first21(&compute_first21(compressed.as_bytes(), ADDRESS_VERSION_BYTE))
    }

    /// The single-block SHA-256 must agree with the general-purpose one
    /// it replaced, including on the padding it hard-codes.
    #[test]
    fn sha256_of_33_matches_crate() {
        let mut rng = rand::thread_rng();
        let mut block = init_sha_block();
        let mut out = [0u8; 32];

        for _ in 0..5_000 {
            let mut pubkey = [0u8; 33];
            rng.fill(&mut pubkey[..]);
            // Real inputs always start with a parity byte; make sure that
            // byte varies rather than being random noise.
            pubkey[0] = if rng.gen::<bool>() { 0x02 } else { 0x03 };

            sha256_of_33(&mut block, &pubkey, &mut out);
            assert_eq!(&out[..], &Sha256::digest(pubkey)[..]);
        }

        // The block is reused across calls, so a shorter-than-expected
        // write could leave stale bytes behind. Hash two very different
        // inputs back to back to catch that.
        for pubkey in [[0x00u8; 33], [0xffu8; 33], [0x02u8; 33]] {
            sha256_of_33(&mut block, &pubkey, &mut out);
            assert_eq!(&out[..], &Sha256::digest(pubkey)[..]);
        }
    }

    /// The multi-buffer RIPEMD-160 must agree with the `ripemd` crate on
    /// every lane, for the 32-byte inputs it is specialized to. Any error
    /// in the round constants, message-word order, rotation amounts or
    /// lane plumbing changes the digest completely, so this is a total
    /// check rather than a spot check.
    #[test]
    fn ripemd160_multi_matches_crate() {
        let mut rng = rand::thread_rng();

        for _ in 0..2_000 {
            let mut inputs = [[0u8; 32]; RIPEMD_LANES];
            for lane in 0..RIPEMD_LANES {
                rng.fill(&mut inputs[lane][..]);
            }

            let mine = ripemd160_multi_32(&inputs);
            for lane in 0..RIPEMD_LANES {
                let theirs = Ripemd160::digest(inputs[lane]);
                assert_eq!(
                    &mine[lane][..],
                    &theirs[..],
                    "lane {} disagrees for input {}",
                    lane,
                    hex::encode(inputs[lane])
                );
            }
        }

        // Inputs that random sampling will not reach, and a case where
        // every lane holds the same value (so a lane-indexing bug that
        // happens to be self-consistent still shows up above, not here).
        let mut edge = [[0u8; 32]; RIPEMD_LANES];
        for (lane, slot) in edge.iter_mut().enumerate() {
            *slot = [[0x00u8, 0xff, 0x80, 0x01][lane % 4]; 32];
        }
        let mine = ripemd160_multi_32(&edge);
        for lane in 0..RIPEMD_LANES {
            assert_eq!(&mine[lane][..], &Ripemd160::digest(edge[lane])[..]);
        }
    }

    /// The multi-buffer path must produce exactly the `first21` the old
    /// one-at-a-time path did, since every pre-filter decision is made
    /// from those bytes.
    #[test]
    fn multibuffer_first21_matches_single_path() {
        let mut rng = rand::thread_rng();

        for _ in 0..500 {
            let mut pubkeys = [[0u8; 33]; RIPEMD_LANES];
            let mut digests = [[0u8; 32]; RIPEMD_LANES];
            for lane in 0..RIPEMD_LANES {
                rng.fill(&mut pubkeys[lane][1..33]);
                pubkeys[lane][0] = if lane % 2 == 0 { 0x02 } else { 0x03 };
                digests[lane].copy_from_slice(&Sha256::digest(pubkeys[lane]));
            }

            let hashes = ripemd160_multi_32(&digests);
            for lane in 0..RIPEMD_LANES {
                let mut multi = [0u8; 21];
                multi[0] = ADDRESS_VERSION_BYTE;
                multi[1..21].copy_from_slice(&hashes[lane]);

                let single = compute_first21(&pubkeys[lane], ADDRESS_VERSION_BYTE);
                assert_eq!(multi, single, "lane {}", lane);
            }
        }
    }

    /// The endomorphism constants must survive their own startup proof.
    #[test]
    fn endomorphism_constants_validate() {
        assert!(
            setup_endomorphism().is_some(),
            "endomorphism constants failed validation — the search would silently \
             fall back to negation-only (correct, but ~1.1x slower)"
        );
    }

    /// The heart of the six-address optimization: for random private keys,
    /// every one of the six (x-variant, parity) combinations must produce
    /// exactly the address that its recovered scalar controls. This
    /// covers the beta/lambda pairing, the lambda^2 case, and the claim
    /// that negating a point flips the parity byte and nothing else.
    #[test]
    fn six_symmetry_variants_match_their_recovered_scalars() {
        let endo = setup_endomorphism().expect("constants must validate");
        let lambda2 = endo.lambda * endo.lambda;
        let mut rng = rand::thread_rng();

        for _ in 0..16 {
            let k = Scalar::random(&mut rng);
            let (x, y) = affine_xy(&(ProjectivePoint::GENERATOR * k).to_affine()).unwrap();
            let y_is_odd = bool::from(y.normalize().is_odd());

            let mut xv = x;
            for vx in 0..3 {
                if vx > 0 {
                    xv = xv * endo.beta;
                }
                let mut compressed = [0u8; 33];
                compressed[1..33].copy_from_slice(&xv.normalize().to_bytes());

                for neg in [false, true] {
                    compressed[0] = if y_is_odd != neg { 0x03 } else { 0x02 };
                    let built =
                        address_from_first21(&compute_first21(&compressed, ADDRESS_VERSION_BYTE));

                    let mut sk = match vx {
                        1 => k * endo.lambda,
                        2 => k * lambda2,
                        _ => k,
                    };
                    if neg {
                        sk = -sk;
                    }

                    assert_eq!(
                        built,
                        address_for_scalar(&sk),
                        "variant vx={} neg={} does not match its recovered key",
                        vx,
                        neg
                    );
                }
            }
        }
    }

    /// All six scalars must be distinct, or the throughput counter would
    /// be inflated by addresses that are really the same candidate.
    #[test]
    fn six_symmetry_variants_are_distinct() {
        let endo = setup_endomorphism().expect("constants must validate");
        let lambda2 = endo.lambda * endo.lambda;
        let mut rng = rand::thread_rng();

        for _ in 0..16 {
            let k = Scalar::random(&mut rng);
            let scalars = [k, -k, k * endo.lambda, -(k * endo.lambda), k * lambda2, -(k * lambda2)];
            for a in 0..scalars.len() {
                for b in (a + 1)..scalars.len() {
                    assert_ne!(scalars[a], scalars[b], "variants {} and {} collide", a, b);
                }
            }
        }
    }

    /// Replays the fused forward/backward batch inversion exactly as the
    /// worker loop runs it, checking both that every recovered inverse is
    /// a true inverse and that the resulting affine addition lands on the
    /// point k256 computes for the same scalar.
    #[test]
    fn fused_batch_generation_matches_k256() {
        let table = build_addition_table(BATCH_SIZE);
        let mut rng = rand::thread_rng();
        let base_scalar = Scalar::random(&mut rng);
        let (px, py) = affine_xy(&(ProjectivePoint::GENERATOR * base_scalar).to_affine()).unwrap();

        let mut products = vec![FieldElement::ONE; BATCH_SIZE];
        let total_inv =
            batch_invert_prefix_products(&table, &px, &mut products).expect("no zero denominators");

        let mut running_inv = total_inv;
        for i in (0..BATCH_SIZE).rev() {
            let (tx, ty) = table[i];
            let d = tx - px;
            let inv_i = if i == 0 { running_inv } else { running_inv * products[i - 1] };
            running_inv = running_inv * d;

            assert_eq!(
                (inv_i * d).normalize(),
                FieldElement::ONE.normalize(),
                "recovered inverse is wrong at index {}",
                i
            );

            let slope = (ty - py) * inv_i;
            let x_new = (slope.square() - px - tx).normalize_weak();
            let y_new = slope * (px - x_new) - py;

            let expected = (ProjectivePoint::GENERATOR
                * (base_scalar + Scalar::from((i + 1) as u64)))
            .to_affine();
            let (ex, ey) = affine_xy(&expected).unwrap();
            assert_eq!(x_new.normalize(), ex.normalize(), "x wrong at index {}", i);
            assert_eq!(y_new.normalize(), ey.normalize(), "y wrong at index {}", i);
        }
    }

    /// The leading-u64 prefix filter must agree with the exact 25-byte
    /// comparison on every input, including inputs deliberately placed on
    /// the boundary words where the shortcut has to defer to it.
    #[test]
    fn fast_prefix_filter_agrees_with_exact_comparison() {
        fn exact(first21: &[u8; 21], b: &PrefixBound) -> bool {
            let mut lo = [0u8; 25];
            lo[0..21].copy_from_slice(first21);
            let mut hi = [0u8; 25];
            hi[0..21].copy_from_slice(first21);
            hi[21..25].copy_from_slice(&[0xff, 0xff, 0xff, 0xff]);
            hi < b.lower || lo >= b.upper_exclusive
        }

        let mut rng = rand::thread_rng();
        let prefixes = ["R", "RC", "RCa", "R9", "RVerus", "RXyz12", "RabcdefghK"];

        for prefix in prefixes {
            let bound = match compute_prefix_bound(prefix) {
                Some(b) => b,
                None => continue,
            };

            for _ in 0..20_000 {
                let mut first21 = [0u8; 21];
                rng.fill(&mut first21[..]);
                first21[0] = ADDRESS_VERSION_BYTE;
                assert_eq!(
                    provably_outside_prefix(&first21, &bound),
                    exact(&first21, &bound),
                    "disagreement for prefix {}",
                    prefix
                );
            }

            // Force the boundary-word case the fast path defers on, plus
            // its immediate neighbours.
            for endpoint in [bound.lower, bound.upper_exclusive] {
                for delta in [0i16, -1, 1] {
                    let mut first21 = [0u8; 21];
                    first21.copy_from_slice(&endpoint[0..21]);
                    first21[20] = first21[20].wrapping_add(delta as u8);
                    assert_eq!(
                        provably_outside_prefix(&first21, &bound),
                        exact(&first21, &bound),
                        "boundary disagreement for prefix {}",
                        prefix
                    );
                }
            }
        }
    }

    /// Regression guard for the specialized Base58 encoder against the
    /// general-purpose crate it replaced on the hot path.
    #[test]
    fn fast_base58_matches_bs58() {
        let mut rng = rand::thread_rng();
        let mut out = [0u8; 34];

        for _ in 0..20_000 {
            let mut addr = [0u8; 25];
            rng.fill(&mut addr[..]);
            addr[0] = ADDRESS_VERSION_BYTE;
            encode_address_base58(&addr, &mut out);
            assert_eq!(
                std::str::from_utf8(&out).unwrap(),
                bs58::encode(&addr[..]).into_string()
            );
        }

        // The payload extremes, which the random sampling above will
        // never reach on its own.
        for fill in [0x00u8, 0xff] {
            let mut addr = [fill; 25];
            addr[0] = ADDRESS_VERSION_BYTE;
            encode_address_base58(&addr, &mut out);
            assert_eq!(
                std::str::from_utf8(&out).unwrap(),
                bs58::encode(&addr[..]).into_string()
            );
        }
    }
}

