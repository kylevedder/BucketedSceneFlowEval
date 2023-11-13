# Bucketed Scene Flow Evaluation

A standardized dataloader plus eval protocol for various scene flow datasets.

## Data Structures:

Located in `datastructures/scene_sequence.py`

### `RawSceneSequence`

`RawSceneSequence` describes the raw scene -- raw observations and their global frame poses.

`RawSceneSequence` presents a map interface from `Timestamp` to `RawSceneItem`.

### `QuerySceneSequence`

`QuerySceneSequence` is a self-contained description of:

 - the raw scene
 - query particles
 - the requested timestamps the prediction method should solve for

Query particles are comprised of a series of particles, each associated with a particle id, and a single query timestamp. The query timestamp associates the particles with the requested timestamps. In principle these particles could be at any point in the requested series, although datasets may provide stronger guarantees (e.g. scene flow datasets will have these be the first of two timestamps)

`QuerySceneSequence` presents a map interface from `ParticleID` to `Tuple[WorldParticle, Timestamp]`.

### `EstimatedParticleTrajectories`

`EstimatedParticleTrajectories` describes trajectories for every `ParticleID` over the given timestamps.

### `EstimatedParticleTrajectories`

`EstimatedParticleTrajectories` describes trajectories for every `ParticleID` over the given timestamps, along with semantic class IDs for each particle.
