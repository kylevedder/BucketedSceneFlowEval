## Data Structures:

Located in `datastructures/scene_sequence.py`

### `RawSceneSequence`

`RawSceneItem` describes the raw scene -- raw observations and their global frame poses.

`RawSceneSequence` presents a map interface from `Timestamp` to `RawSceneItem`, where the `Timestamp` is an `int` representing the timestep in the sequence that this observation was collected at.

### `QuerySceneSequence`

`QuerySceneSequence` is a self-contained description of:

 - the raw scene
 - query particles
 - the requested timestamps the prediction method should solve for

Query particles are comprised of a series of particles, each associated with a particle id, and a single query timestamp. This object effectively describes a flow or point tracking problem. Given a query point(s) and a start time, provide motion vectors for the requested (future) timesteps. In principle the query particles could begin at any point in the requested series, although datasets may provide stronger guarantees (e.g. scene flow datasets will have the query as the first of two timestamps).

`QuerySceneSequence` presents a map interface from `ParticleID` to `tuple[WorldParticle, Timestamp]`.

### `EstimatedPointFlow`

`EstimatedPointFlow` describes trajectories for every `ParticleID` over the given timestamps.

### `GroundTruthPointFlow`

`GroundTruthPointFlow` describes ground truth trajectories for every `ParticleID` over the given timestamps, along with semantic class IDs for each particle.
