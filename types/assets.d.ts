/**
 * Side-effect CSS imports in components. A per-file `declare module './X.css'` does not
 * work: TypeScript treats a relative specifier in a module augmentation as a reference to
 * an existing module, so the declaration must be a wildcard in an ambient file.
 */
declare module '*.css';
