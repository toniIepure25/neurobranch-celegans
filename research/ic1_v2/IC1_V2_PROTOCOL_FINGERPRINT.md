# IC1-v2 protocol fingerprint

The [estimand specification](IC1_V2_ESTIMAND_SPEC.md) was written after reproducing the historical defect. At 12:15:38 on 25 November 2026 in Bucharest, before adding the v2 implementations or running the frozen grid, its bytes were fingerprinted. The preimplementation canonical SHA-256 is:

```text
b52c929c6be758b1b47304f238b7ea574e4ebbc689d74193fa2bf94faa45ef92
```

The acceptance runner pins this digest. Canonicalization decodes strict UTF-8, converts CRLF and bare CR to LF, and encodes UTF-8. It preserves all other bytes under that text representation. The test checks LF, CRLF and bare CR equivalence and changes 8192 to 4096 to test identity sensitivity. The calibration JSON records the canonical specification digest, the actual input byte digest and separate raw-byte source digests. The local timestamp and pinned bytes record this execution sequence.

Historical IC1 used a raw-byte hash. Its recorded CRLF digest remains `37eba808bc549a394cdc7673eed09ecda01e000f2a66ef9bb8aec18d78abef1f`. The LF materialization hashes to `e15f82e27c8a35badea11e5a57e254d23ddf653f457f1c84be8bd2e1779c1c1d`. The original protocol, raw-byte hash, and output manifest are unchanged. [Historical protocol](../interventional_consistency_protocol.md) · [Historical raw digest](../interventional_consistency_protocol.sha256) · [Calibration results](IC1_V2_ACCEPTANCE_RESULTS.json)
