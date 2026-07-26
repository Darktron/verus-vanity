# verus-vanity
## Veruscoin Vanity Wallet Generator

1. Get Rust:
```
sudo apt update && sudo apt install rustc
```

or

```
pkg update && pkg install rust
```

2. Clone the repo:
```
git clone https://github.com/Darktron/verus-vanity.git
```

3. Build:
```
cd ~/verus-vanity && RUSTFLAGS="-C target-cpu=native" cargo build --release
```

or


```
cd ~/verus-vanity && cargo build --release
```

4. Move the binary:
```
mv ~/verus-vanity/target/release/verus-vanity ~/verus-vanity/
```

5. Use example:
```
~/verus-vanity/verus-vanity -m 1 -p RDARK -o wallets.txt
```

6. Help & options:
```
~/verus-vanity/verus-vanity -h
```

```
VerusCoin Vanity Wallet Generator

Usage: verus-vanity [OPTIONS]

Options:
  -p, --prefix <prefix>    Prefix string or filename with prefixes (one per line). 'R' is added automatically if omitted.
  -s, --suffix <suffix>    Suffix string or filename with suffixes (one per line)
  -m, --matches <matches>  Number of matching addresses to find; -1 for infinite [default: -1]
  -t, --threads <threads>  Number of threads (default = number of CPU cores) [default: 12]
  -o, --output <output>    Output file to save found wallets
  -h, --help               Print help
  -V, --version            Print version
```

### Affixes:
Prefix: `Crypt` (Start)
Infix: `ograph` (Middle)
Suffix: `y` (End)
All: `Cryptography`

verus-vanity v0.2.0 performance achieved 950,000 - 1,250,000 wallets per second (0.95 - 1.25MW/s) on a Snapdragon 8 Elite Oryon (2× 4.32 GHz Prime cores + 6× 3.53 GHz Performance cores) all modes

verus-vanity v0.3.0 performance achieved 2,850,000 - 3,150,000 wallets per second (2.85 - 3.15MW/s) on a Snapdragon 8 Elite Oryon (2× 4.32 GHz Prime cores + 6× 3.53 GHz Performance cores) all modes

verus-vanity v0.3.0a performance achieved 6,500,000 - 8,100,000 wallets per second (6.50 - 8.10MW/s) on a Snapdragon 8 Elite Oryon (2× 4.32 GHz Prime cores + 6× 3.53 GHz Performance cores) --prefix only

verus-vanity v0.4.0 performance achieved 8,880,000 - 10,550,000 wallets per second (8.88 - 10.55MW/s) on a Snapdragon 8 Elite Oryon (2× 4.32 GHz Prime cores + 6× 3.53 GHz Performance cores) --prefix only
