# Models and Numerical Conventions

## Crowd Geometry and Physical Time

Coordinates are zero-based (x,y), with x rightwards and y downwards. Arrays use
[time,y,x]. Walls are grey; red crosses are attraction locations, not absorbing
exits. The 15x15 concert room has twenty equally weighted starting cells and
attraction locations (1,14) and (13,14), both on its bottom boundary wall. Those
points influence preferences without being reachable cells. The asymmetric
7x7 maze starts at (1,1) and has the reachable attraction point (5,5).

The initialization adapter probes the pinned C++ game's coordinate convention,
adjusts initial-position encoding and checks actual chance outcomes. Walls and
targets are never transposed. On the legacy initialization implementation both
the requested initial support and its transpose must avoid walls; unsupported
starts fail explicitly. Raw wall mass is checked before masking for display.

Library horizon H gives decision slices 0,...,H-1, followed by a zero-payoff
terminal state at H. Concert-room data use H=21 and maze data H=11. An algorithm
update is a different index from physical time. The last outgoing physical move
is processed by the native game but contributes no later nonzero reward.

## Crowd Reward, Noise and Local Value Correction

With grid size d, attraction points g_j and weights b_j, the native decision
reward is

    u_t(s, m) - ||a_previous||_1/d,
    u_t(s, m) = 2 - (2/d) sum_j b_j ||s-g_j||_1 - c log(m_t(s)+epsilon).

Here epsilon is the single-precision constant near 1e-25. All examples use
`with_congestion=false` and `only_distribution_reward=false`: density does not
multiply the movement cost. The positional penalty uses the weighted sum of
Manhattan distances, not the distance to the nearest target. With equal weights
at the two concert-room points it is horizontally constant between x=1 and 13.

At initialization, the native reward uses uniform density on all d*d cells and
previous action stay. This initial contribution is action independent. Later
rewards use the supplied population distribution. Intended actions are legal
free-neighbor moves or stay. At the post-action position, noise is stay with
probability 1-p and is uniform over legal actions there, including stay, with
probability p. It is not unrestricted additive cardinal noise followed by one
combined rejection. Boundary walls prevent wrap-around in the supplied layouts.

The preceding action is absent from native position/time observation keys.
Caching raw values by those keys can therefore merge different pending movement
costs. The local normalized backend stores the history-independent quantity W
in V(history)=W(observation)-pending_cost(history), and restores the offset on
retrieval, including in action values. It preserves pathwise total reward and
physical transitions. A separate observable action-cost-node adapter provides
an independent reference formulation. Equivalently, movement costs can be paid
immediately for decisions 0,...,H-2, with positional/density reward only at H-1.

Corrected runs use `reward_mode=markov_action_cost_v1` and
`value_backend=normalized`. FP, fixed point and NashConv use scoped local value
bindings; these are restored after the run. OMD and BPI use local subclasses.
OMD freezes its cumulative-score table so each backward pass evaluates one
fixed previous policy. No installed Python or C++ source is edited. Do not run
overlapping normalization scopes in different threads of one Python process.
Only the documented population-independent movement-cost configuration is
supported by these corrections.

Analytical tests, uncached short-history tests and independent final-policy
replays support the recorded metrics for these configurations. They do not
certify all upstream games or evaluators. The twenty concert-room initial
weights are stored in single precision and sum to 1.0000000149011612; they are
not silently renormalized. Probability checks allow the corresponding rounding.

## Learning Methods and Saved Policies

All methods start uniformly on legal actions, recorded at update zero. The
concert room has 100 updates; the maze has 50. Eight curves cover seven families:

| Method | Update / parameter |
|---|---|
| Fixed point | Greedy best response to current induced flow |
| Softmax fixed point | Optimal-Q soft response, temperature 0.1 |
| Fictitious play | Occupancy-weighted merge, new-response weight 1/(k+1) |
| Damped fixed point | `FictitiousPlay(lr=0.01)`, constant merge weight 0.01 |
| Softmax fictitious play | Decreasing merge weight, optimal-Q temperature 0.1 |
| OMD | Softmax of cumulative evaluated Q-functions, step size 0.1 or 0.01 |
| BPI | Current evaluated-Q softmax, inverse temperature 0.1 |

The merge weights policies by their induced state occupancies, not pointwise.
Because these crowd transitions are independent of density, it induces the
corresponding average of flows. Ordinary FP also averages the initial uniform
policy's occupancy. This flow identity is not asserted for general
distribution-dependent dynamics. BPI denotes this implementation's evaluated-Q
softmax policy iteration, not an assertion of equivalence to a separately
regularized best-response formulation.

Both metrics and flows use the returned `get_policy()` policy. Curves are raw
unnormalized NashConv; a final iterate is not automatically an equilibrium.
Tabular computations sum over transition probabilities rather than drawing
population sample paths. High exploitability values/oscillation are not Monte
Carlo noise. Saved action-probability tables use representative native
information-state histories; the verifier maps them to physical observations
for independent replay. They are not resumable learning checkpoints.

## Aligned Traffic Model

The links are A->B, B->C, B->D, C->D, C->E, D->E, E->F. Origin and destination
are links A->B and E->F; node D is not the destination. Time step Delta=0.25;
all link capacities are 4. Free-flow times in that order are
(0,1,2,0.25,2,1,0). BPR time is t0(l)[1+a(l)*(v/4)], with a=1 on B->C and
D->E and zero elsewhere. Demand grows with N at fixed capacity.

Finite congestion uses realized vehicle counts. Mean-field congestion uses
N times post-routing link probability, summed over all waiting states.
An entered link receives max(0,floor(travel_time/Delta)-1) waiting steps.
Positive waits decrement before another move becomes available; absorbed and
waiting states have only a dummy action. A sentinel -1 represents an unassigned
wait during native bookkeeping. No additional travel noise is used.

The local `fixed_horizon_v1` subclasses process exactly H=17 physical updates
in both games. Arrived and wrong-sink vehicles stay absorbed to H. Both games
give nonarrival payoff -H*Delta=-4.25. Arrival during the action from index t
to t+1 retains native payoff -t*Delta. This is not literally an unqualified
per-step -1 reward. Destination populations remain visible in the saved flow.
The native finite running-reward interface is locally aligned with terminal
returns, including the truncated last step; MCCFR uses terminal returns.

Waits at least as large as the remaining horizon are represented by that
remaining horizon. This does not change within-horizon movement or arrival,
and keeps busy mass inside native mean-field support at high demand. Local OMD
also snapshots cumulative scores before evaluating its previous policy. These
local adapters do not make finite and mean-field policies/payoffs identical
in general. The implementation and tests deliberately distinguish the games.

The population illustration uses N=4, ten OMD updates and step size 1, with
saved indices 0/1/8/10/16 (physical times 0/.25/2/2.5/4). Frozen score keys and
values in `population.npz` restore the policy through
`traffic_aligned.restore_policy`; generic history-keyed tabular conversion
is not used when the distribution changes.

## Timing Protocol

The grid is N=2,5,10,15,20,25,30,40,45,50 with 20 updates and three repetitions
per solver and population. CFR is external-sampling MCCFR with SIMPLE averaging;
one iteration updates every player. OMD updates one representative-agent policy.
Fresh child processes run sequentially with one numerical-library thread,
one untimed warm-up update per solver, then fresh measured solvers. Seed is
7+1000*N+r for r=0,1,2. Solver order alternates by full-grid index and repetition.
Three full-budget pilot pairs were reused with identical implementation/settings
protocol; there are thirty distinct pairs, not thirty plus three. Native-model
timings and older five-repetition timings are not included.

Only iteration-loop time is plotted. Construction, startup, warm-up, metric
evaluation and plotting are excluded. Error bars are sample SD (ddof=1), not
confidence intervals. Separate conventional per-point Student-t mean intervals
assume independent approximately normal timings; three observations cannot
verify those assumptions. At these settings maximum relative half-widths were
9.89% for CFR and 11.18% for OMD. No observations were discarded.

This compares fixed update budgets, not equal work or time to equal accuracy.
High demand can leave journeys unfinished at the deadline. OMD is deterministic
here; repetition spread measures timing variation, whereas MCCFR also samples
traversals. Absolute seconds depend on hardware/load. Portable provenance
records packages, platform, thread limits and numerical-run source fingerprints;
precise CPU-model/RAM information was unavailable.

## Figures

Distribution columns are physical-time slices under each final policy, not
learning iterates. Heatmaps labelled "Population density" show probability mass
per grid cell, with one common scale per grid across selected methods and all
saved times, including initialization. No panel is renormalized or clipped.
Logarithmic curves alone display max(value,1e-12); saved values are untouched.
Traffic circle radius, not area, is proportional to link probability.

Native heatmap cells and uncompressed PDF streams avoid a low-bit-depth export
problem in the pinned Matplotlib backend. Fonts and labels should still be
reviewed at the manuscript's final printed size.
