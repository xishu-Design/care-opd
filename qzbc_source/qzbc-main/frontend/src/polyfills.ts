type StructuredCloneCompat = <T>(value: T, options?: { transfer?: unknown[] }) => T;

const scope = globalThis as typeof globalThis & { structuredClone?: StructuredCloneCompat };

if (typeof scope.structuredClone !== "function") {
  scope.structuredClone = <T>(value: T): T => clonePlainValue(value, new WeakMap<object, unknown>());
}

function clonePlainValue<T>(value: T, seen: WeakMap<object, unknown>): T {
  if (value === null || typeof value !== "object") return value;
  if (seen.has(value)) return seen.get(value) as T;
  if (value instanceof Date) return new Date(value.getTime()) as T;
  if (value instanceof RegExp) return new RegExp(value.source, value.flags) as T;
  if (value instanceof Map) {
    const cloned = new Map();
    seen.set(value, cloned);
    value.forEach((item, key) => cloned.set(clonePlainValue(key, seen), clonePlainValue(item, seen)));
    return cloned as T;
  }
  if (value instanceof Set) {
    const cloned = new Set();
    seen.set(value, cloned);
    value.forEach((item) => cloned.add(clonePlainValue(item, seen)));
    return cloned as T;
  }
  if (Array.isArray(value)) {
    const cloned: unknown[] = [];
    seen.set(value, cloned);
    value.forEach((item) => cloned.push(clonePlainValue(item, seen)));
    return cloned as T;
  }

  const prototype = Object.getPrototypeOf(value);
  // Cytoscape layout options and elements are plain objects. Keep browser and
  // library class instances by reference instead of corrupting their prototype.
  if (prototype !== Object.prototype && prototype !== null) return value;
  const cloned: Record<PropertyKey, unknown> = Object.create(prototype);
  seen.set(value, cloned);
  Reflect.ownKeys(value).forEach((key) => {
    cloned[key] = clonePlainValue((value as Record<PropertyKey, unknown>)[key], seen);
  });
  return cloned as T;
}
