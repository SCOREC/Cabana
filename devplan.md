## dev steps

1. indexing interface
2. particle movement between elements
3. sorting
4. cabana access functions

### Indexing Interface

Assuming we had a fixed number of particles per element and we were using
Cabana as it exists now, the parallel element and particle loops would look
like the following:

```
parallel_for(aosoa.numSoa) { // array of element structs
  //execute mesh element calculations
  parallel_for(aosoa.arraySize(s)) { // particle arrays within elm
    //apply values from the element calculations to each particle
  }
}
```

Our indexing layer would replace the outer loop over the aosoa since multiple
soa could be needed to represent a single element.
To facilitate a loop over
elements an array of soa offsets could be used to track which soa are assigned
to each element.
For example, if there is one soa for element one, four for element two, and
two for element three, then we would store a prefix-summed offset list:

```
offsets = [0, 1, 5, 7]
```

where the number of soa for element i (zero based indexing) can be found with
`offsets[i+1] - offsets[i]`.


```
parallel_for(numElements) {
  numSoa = offsets[i+1] - offsets[i]
  //execute mesh element calculations
  parallel_for(numSoa) {
    parallel_for(aosoa.
arraySize(s)) { // particle arrays within elm
      //apply values from the element calculations to each particle
    }
  }
}
```

In the indexing layer we would store the `offsets` array and additional arrays
to support reassigning a given particle to a different element without
rebuilding the full AOSOA.

Tasks:
- extend/derive the Cabana AOSOA to hold an offsets array, referring to it
  below as the ElmAOSOA
- define a mechanism to run a parallel loop over the elements (as done in the
  pseudo code above)
- assume that all particles are active (i.e., no extra capacity)

### Particle Movement

As particles are 'pushed' through the domain their parent element will change.
Rebuilding the entire AOSOA because a few of the million or more particles have
changed elements may be a prohibitive cost in terms of memory and runtime.
Thus, we should support changing the assignment of a particle to an element.

First, we assume that a user can control the amount of extra capacity each
element has for particles.
Thus, there will be used and free particles (tuples).
To track this, each element stores the position of the first free particle that
belongs to each element.
All indices prior to this position are used, and all after it, including the
index itself, are free.

Given an initial distribution of particles, we allocate a larger than
necessary set of tuples, and reserve spare tuples at the end of each elements
range of active tuples.
When a particle changes element we mark the source tuple as spare and write the
data to the first spare tuple in the elements range.
Note, a supporting function will have to be written to support this permutation
of particles.
When there are no spare tuples for a given element, the structure could either
be rebuilt with additional capacity, or the spare tuples redistributed (via
sorting) to place them in the range of the elements that require them.

The particle permutation will collect the lists of particles to move (three
steps: 1) count particles that are moving 2) allocate CSR 3) write to CSR) and
then write the changes to the structure.
Any holes created in an SOA from particles leaving an element should be filled
immediately by swapping in a particle from the end of the SOA into the hole.
Note, if the majority of particles are moving the cost will approach that of a
full rebuild.

Tasks:
- extend the ElmAOSOA to hold a first free index
- implement the permutation function - assume the user will provide a numParticles
  sized array to store each the id of each particles next parent element - could
  this be built-in?

### Sorting and Rebuilds

#### Degree Skew 

It seems unlikely that sorting the AOSOA by element degree will improve
performance or memory usage in Cabana.
In the SCS structure, a Chunk is a contiguous array of memory sized
ChunkSize x maxElementDegreeInChunk.
So, if there was one relatively high degree element in a chunk, then a 
significant amount of space would be wasted.
Sorting all the elements by degree minimizes the amount of wasted space
by grouping elements with similar degree together in chunks.

If I understand correctly, in Cabana the SOAs that comprise one of our elements
are sized to be SIMDwidth x tupleMembersSize.
Since the entire SOA will be holding data for a single element, then there is no
degree skew problem.

#### Large Changes in Particle Distribution

After a large number of particle push operations the distribution of particles
to elements could change significantly; i.e., elements that initially have a
high degree could become empty and vice versa.
Thus, a previously high degree element will have many SOAs that are unused that
could be reassigned to newly high degree elements.
A permutation function could support the reassignment such that a rebuild is not
required.

Tasks:
- define a helper function that uses the permutation function to attempt to swap
  an element with a lot of unused capacity with one that is out of space - this
  could be done periodically for the worst elements to delay/prevent a full
  rebuild
- define a rebuild function that creates a new ElmAOSOA with sufficient capacity
  to hold values from an existing ElmAOSOA

### Cabana Access Functions

The indexing layer will likely be presented to users via new Cabana APIs.
The use of existing access APIs, such as `slices`, needs to be supported if
possible.
If I understand correctly, a `slice` is a Kokkos View with a defined stride.

Tasks:
- if possible, provide a `slice` from an ElmAOSOA that accounts for used and free
  particles/tuples
- determine which other Cabana interface APIs should be supported by ElmAOSOA
