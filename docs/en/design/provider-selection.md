# Operator Provider Selection

Sparse-vLLM resolves an operator exactly once during model construction:

```text
OpSpec
  -> atomic capability filter
  -> exact local profile overlay
     -> matched: bind the profiled dispatch plan
     -> unmatched: apply the default portfolio policy
  -> prepare the selected implementation
  -> execute without resolver re-entry or silent fallback
```

## Atomic Capabilities

`supports()` answers only whether an atomic implementation can correctly satisfy
the operation contract on the active platform. It returns one typed status:

- `SUPPORTED`: the implementation is correct for the contract.
- `UNSUPPORTED_CONTRACT`: normal semantic or platform mismatch.
- `DEPENDENCY_ABSENT`: an optional upstream family is not installed.
- `DEPENDENCY_BROKEN`: an installed dependency has an incompatible version,
  ABI, or callable contract.

Local benchmark coverage is not an atomic support status. In particular, an
upstream implementation must not reject a shape merely because Sparse-vLLM did
not benchmark that shape locally.

## Portability Of Repository-Owned DSL Kernels

`repo_nonstandard` describes ownership of a nonstandard operation contract; it
does not imply a hardware-specialized implementation. For repository-owned
kernels written with portable Triton or TileLang primitives, keep the atomic
support domain broad. Derive it from semantics, tensor contracts, the DSL and
toolchain, hardware features actually used, and known compiler limitations.
Missing local validation for a GPU model or shape is not by itself a reason for
a device-name whitelist, compute-capability whitelist, or atomic rejection.

Sparse-vLLM is a research-oriented project and cannot pre-validate every
hardware combination. On a device with no known incompatibility, the resolver
may optimistically bind a portable DSL provider and attempt compilation during
prepare, JIT, warmup, or first execution. Compilation or execution failures must
surface the original actionable error. They must not trigger silent provider
reselection, fabricated default outputs, or masked failures. After an
incompatibility is confirmed, prefer an exclusion expressed by the required
hardware feature, DSL capability, or known toolchain issue over a permanent
device-model whitelist.

Keep atomic eligibility separate from validation and performance evidence.
`validation_evidence` records only the devices, shapes, dtypes, graph modes, and
results actually tested; missing evidence does not automatically narrow a
portable support domain. In contrast, performance profiles, default performance
preferences, and cross-system performance claims must stay within reproducible
measured evidence. A kernel may be eligible to run broadly while claiming
validated correctness or superior performance only where evidence exists.

Apply an **optimistic portability, conservative evidence** rule to
repository-owned nonstandard kernels. When a nonstandard contract is implemented
with portable Triton or TileLang primitives and has no known incompatibility, its
general atomic provider should remain eligible beyond the small set of locally
available GPUs. If that provider is the normal implementation of the contract,
place it in the appropriate default portfolio. Limited local hardware coverage
belongs in `validation_evidence`; it must not by itself turn the whole provider
into an exact-device profile.

Reserve `profile_only=True` for a genuinely specialized alternative: for example,
an exact-device launch schedule, a measured token-range dispatcher, or a mixed
provider plan that should override the general provider only inside its recorded
performance domain. Prefer the following structure:

```text
nonstandard operation contract
  -> general portable Triton/TileLang atomic provider in the default portfolio
  -> optional exact profile selecting a tuned provider, schedule, or dispatch plan
```

Do not use an exact profile as a substitute for a portable default merely because
the repository cannot test many GPU models. Profile misses must retain a valid
implementation of the nonstandard contract whenever such an implementation is
known to be portable.

## Default Portfolio

Every operator registry owns an explicit `PortfolioPolicy`. Standard upstream
providers are listed before repository portable baselines. Repository-owned
nonstandard providers are used for sparse scores, cache layouts, state mutation,
or other contracts that upstream standard operators cannot express. Provider
classes do not declare integer priorities.

An atomic provider omitted from the default portfolio must be registered with
`profile_only=True`. This is reserved for a specialized implementation referenced
by an exact profile; accidental hidden providers fail registry validation. The
flag controls default selection eligibility, not the provider's correctness
support domain or the scope of local validation evidence.

## Profile Overlays

Profiles live in a separate registry. A profile declares the atomic providers it
uses, matches device, shape, operation contract, and required toolchain, and
builds a prepared dispatch plan. The resolver checks atomic correctness before
calling the profile matcher. A profile miss has no effect on atomic eligibility
or on the default portfolio.

Profile precedence is an explicit registry-level order. A profile may override a
default performance choice, but it must never define the support domain of a
standard upstream operator or disable a portable repository-owned nonstandard
path merely because local performance data is absent.

## Phase Composition

A semantic operator may compose independently selected execution phases. Full
attention is owned by one prepared `FullAttentionProvider`, while prefill and
decode keep separate atomic registries because their kernels, workspaces, graph
contracts, and support domains can differ. A prefill-only implementation such
as FlexPrefill participates only in the prefill portfolio and does not need a
decode implementation.

The full-attention provider validates the shared head, dtype, scale, causal,
page-layout, and page-table contract before preparing either phase. It then
binds both prepared phase operators to the model as one lifecycle and closes
them together. Phase selection remains independent, so hybrid upstream
prefill/decode pairs are valid when their shared cache contract matches.

## Dependency And Evidence Rules

If an applicable upstream dependency is absent, the resolver may bind a
repository baseline and records `selection_basis=dependency_degraded`. A broken
installed dependency fails binding. Runtime exceptions never trigger provider
reselection.

Every binding report records the selected provider and optional profile,
`selection_basis`, all atomic and profile decisions, and `validation_evidence`.
An unprofiled upstream selection therefore states that adapter equivalence was
validated, kernel support is upstream-declared, and performance relies on the
upstream default rather than a local benchmark.

Evidence follows ownership. `adapter_equivalence` and `upstream_declared` are
used only for a purely upstream atomic path. Portable baselines, repository
nonstandard kernels, and profiles that mix atomic roles report their narrower
contract evidence instead of inheriting upstream validation claims.

## Ownership Rule

For standard operations, prefer upstream atomic providers and maintain only the
adapter plus a portable repository baseline. Add a repository-owned production
kernel only for new sparse semantics or a runtime contract that upstream cannot
express. Local profiles can override default selection; they cannot narrow
upstream support. Prefer portable Triton or TileLang implementations for
repository-owned nonstandard kernels: their semantics may be nonstandard, but
limited local hardware access must not artificially narrow their hardware
support domain.
