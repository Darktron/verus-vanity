# verus-vanity
## Veruscoin Vanity Wallet Generator

1. Get Rust:
```
sudo apt update && sudo apt install rustc git -y
```

or

```
pkg update && pkg install rust git -y
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
  -b, --batch <batch>      Points per batch (min = RIPEMD lane count; max scales with -t and RAM)
  -e, --eta <eta>          Seconds between ETA refreshes; 0 shows it once only [default: 30]
  -S, --serve <serve>      Run as cluster master on ADDR:PORT (also searches locally)
  -J, --join <join>        Run as cluster worker, taking the objective from the master at ADDR:PORT
  -P, --pass <pass>        Shared word a worker must present to join a cluster [default: ]
  -N, --name <name>        Name this worker reports to the master [default: hostname-ish]
  -k, --keepalive          Worker: stay running when the master stops, and wait for the next objective
      --stop-workers       Master: tell workers to exit when the objective is met, even keepalive ones
  -d, --dismiss <dismiss>  Shut down every worker that connects to ADDR:PORT, then exit
      --keys-stay-local    Worker keeps found keys on its own machine; only addresses are sent
  -B, --bench              Measure per-stage throughput on this machine and exit
  -v, --version            Print version
  -h, --help               Print help
```

### Affixes:
Prefix: `Endo` (Start)

Infix: `morph` (Middle)

Suffix: `ism` (End)

All: `Endomorphism`
