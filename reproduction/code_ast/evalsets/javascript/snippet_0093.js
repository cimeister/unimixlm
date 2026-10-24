// union of two sets
Set.prototype.union = function (set) {
    return new Set([...this, ...set]);
};

// intersection of two sets
Set.prototype.intersect = function (set) {
    return new Set([...this].filter(element => set.has(element)));
};

// difference of two sets
Set.prototype.diff = function (set) {
    return new Set([...this].filter(element => !set.has(element)))
};