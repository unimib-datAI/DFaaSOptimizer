# Logical communication accounting, version 1

Enable `communication_metrics: true` in the solver configuration. Supported
methods are MADEA-detailed, MADEA-one-shot-detailed, MADEA-selective,
MADEA-one-shot-selective, FaaS-MAPG-R-detailed and Greedy-RR. Unsupported methods
fail explicitly instead of publishing partial totals as complete measurements.

The implementation simulates nodes in one process/shared memory. These counters
record the algorithm's executed exchanges under **dfaas-logical-batched-v1**,
an explicit communication model. They do not measure real packets, serialized
bytes, routing hops, SSH/Ray traffic or application request forwarding. Claims
in a paper must name this model, rather than call the results measured network
traffic. In particular, state advertisements and PG queries/turns below define
how shared-state accesses would be realized as messages.

## Counting convention

One message is one delivery from a sender to a receiver in a protocol phase.
All function records in that phase are batched for that node pair. The reverse
direction is a different delivery. Repeated rounds/PG visits count again.
No transport retransmissions or MTU fragmentation are assumed. Self-access is
local. A broadcast counts once per recipient, not once per transmitter.

Unit auctions do not imply one network message per unit bid. Repeated capacity
offers with the same sender, receiver, function, bid price and utility are
encoded as one record carrying their summed quantity. For example, 1000 equal
unit offers are one record with quantity 1000, in one message to that receiver.
Different prices/utilities remain distinct records in that same message. The
receiver can expand the compact representation for its local auction; accounting
does not change the actual auction decisions or switch `unit_bids` off. Offers
from different rounds cannot be merged because later offers depend on replies.

`items` counts typed records in these batches, not bytes or scalar fields.
`request_units` counts the workload quantity represented by capacity offers or
the absolute quantity changed by PG commits; repeated offers count again. It
is not the number of successfully served requests. Memory requests do not
acquire a quantity field merely for accounting: their `request_units` is zero.

| Kind | Deliveries and records |
|---|---|
| `state_advertisement` | MADEA variants and Greedy: each round, each potential receiver sends its state to every eligible source (reverse of the offload edge). One record per function plus one memory record; a function record includes price for auctions. Full state is advertised even when unchanged. |
| `capacity_request` | Actual bids/requests submitted to the capacity phase, batched per source/destination. Items are compact offer records (identical unit offers merged as above), or Greedy function requests; units are the sum of offered quantities. |
| `capacity_reply` | One reply batch to every bidder, including rejections. One result record per function. MADEA additionally notifies displaced incumbents; these records and bidder replies are merged per receiver/source pair in the same phase. |
| `memory_request` | Only requests passed to the memory allocation phase, after confirmation and selective filtering where applicable. Items retain `(i,j,f)`. PG requests are counted when submitted after the sweep, including requests whose receiver no longer has free memory. |
| `pg_turn` | One logical turn grant per node visit, including visits yielding no move. This is a separately reported scheduling/control cost, not a measured token-routing protocol. |
| `pg_state_query` | For every PG node visit, one query to each outgoing neighbor. One item per query. |
| `pg_state_reply` | One state reply from each queried neighbor: one record per function plus one memory snapshot record. Capacities reflect the mover's releasable assignments; memory follows the sweep snapshot used by the implementation. |
| `pg_commit` | Only accepted moves: one batch per destination with changed offloaded flows, including releases. One item per changed function; units are the sum of absolute changes. Local-only changes produce no commit message. |

Memory allocation has no separate acknowledgment in this convention: new
availability is exposed by the next state advertisement/query. PG commits need
no additional acknowledgment under the serialized-turn assumption. Initial
topology, own beta/latency and function descriptors are assumed known. Global
termination detection, barriers, topology discovery, routing and failure recovery
are not implemented distributed protocols here and are excluded for all methods.
When comparing coordination costs, report PG control separately from data
messages; do not interpret it as a complete network implementation overhead.

## Artifacts and timing

`communication.csv` has `step,round,kind,messages,items,request_units`, with
zero-based rounds and physical step indices. Rows aggregate repeated exchanges
of the same kind within that round. Empty categories are omitted; an idle run
still gets a header and zero totals.

`communication.json` records the model/schema version, method, completion flag,
totals, and separate `data_messages`/`control_messages`. The campaign harness
copies this summary to `out/result.json` and requires complete artifacts when
metrics are enabled. Failures may leave partial CSV/JSON with `complete: false`;
these must not enter successful-job comparisons.

Counters are run-scoped observers, independent of verbosity, workers and CUDA.
Their work is inside `step_time.untimed()`, including building record sets, and
does not enter modelled `algorithm_s`. Raw wall time still includes observer and
serialization overhead. They do not alter decisions, random-number generation,
accepted flows, stopping rules or data available to algorithm participants.

For comparisons, use the same instances, budgets and instrumentation version.
Older results without these artifacts cannot supply exact totals retrospectively.
