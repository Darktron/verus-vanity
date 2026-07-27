# verus-vanity
## Veruscoin Vanity Wallet Generator

1. Get Rust:
```
sudo apt update && sudo apt install rustc git
```

or

```
pkg update && pkg install rust git
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
~/verus-vanity/verus-vanity -m 1 -p RVerus -o wallets.txt
```

6. Help & options:
```
~/verus-vanity/verus-vanity -h
```

```
VerusCoin Vanity Wallet Generator
Made by Darktron

Usage: verus-vanity [OPTIONS]

Options:
  -p, --prefix <prefix>    Prefix string or filename with prefixes (one per line)
  -i, --infix <infix>      Infix string or filename with infixes (one per line)
  -s, --suffix <suffix>    Suffix string or filename with suffixes (one per line)
  -m, --matches <matches>  Number of matching addresses to find; -1 for infinite [default: -1]
  -t, --threads <threads>  Number of threads (default = number of CPU cores) [default: 12]
  -o, --output <output>    Output file to save found wallets
  -v, --version            Print version
  -h, --help               Print help
```

### Affixes:
Prefix: `Endo` (Start)

Infix: `morph` (Middle)

Suffix: `ism` (End)

All: `Endomorphism`
