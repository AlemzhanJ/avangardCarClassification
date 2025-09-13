"use client";

import React from "react";

type Props = {
  active: boolean;
  /** One sweep duration in ms */
  durationMs?: number;
  /** Base opacity of the white cover [0..1] */
  baseOpacity?: number;
};

/**
 * ShimmerSkeletonOverlay
 * A subtle white skeleton loader with a sweeping shimmer.
 * Place inside a `relative` container; it will absolutely cover it.
 */
export default function ShimmerSkeletonOverlay({ active, durationMs = 1400, baseOpacity = 0.68 }: Props) {
  if (!active) return null;

  return (
    <div
      className="skeleton-overlay"
      style={{
        "--shimmer-duration": `${durationMs}ms`,
        "--skeleton-base-opacity": baseOpacity,
      } as React.CSSProperties}
    >
      <div className="skeleton-base" />
      <div className="skeleton-sheen" />
    </div>
  );
}
