"use client";

import React from "react";
import { CircularProgressbarWithChildren, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

type Props = {
  label: string;
  value: number; // 0..1
  // If true: low (0) -> green, high (1) -> red
  riskScale?: boolean;
  Icon: React.ComponentType<{ size?: number; weight?: any; className?: string }>;
  className?: string;
};

export default function GaugeRadial({ label, value, riskScale = true, Icon, className = "" }: Props) {
  const v = Math.max(0, Math.min(1, value || 0));
  const color = riskScale
    ? v > 0.66 ? "#ef4444" : v > 0.33 ? "#f59e0b" : "#22c55e"
    : v > 0.66 ? "#22c55e" : v > 0.33 ? "#f59e0b" : "#ef4444";

  return (
    <div className={`p-4 rounded-xl bg-card-background flex flex-col items-center ${className}`}>
      <div className="text-3xl sm:text-4xl font-black text-foreground mb-6">{label}</div>
      <div className="relative w-full" style={{ maxWidth: 320 }}>
        <div style={{ width: "100%", height: 160 }}>
          <CircularProgressbarWithChildren
            value={v * 100}
            maxValue={100}
            circleRatio={0.5}
            styles={buildStyles({
              rotation: 0.75, // start at 180deg
              pathColor: color,
              trailColor: "#e5e7eb", // gray-200
              strokeLinecap: "round",
              pathTransitionDuration: 0.6,
            })}
          >
            <div style={{ transform: 'translateY(-24px)' }}>
              <Icon
                size={96}
                weight="bold"
                className={
                  riskScale
                    ? (v > 0.66 ? 'text-red-500' : v > 0.33 ? 'text-yellow-400' : 'text-green-500')
                    : (v > 0.66 ? 'text-green-500' : v > 0.33 ? 'text-yellow-400' : 'text-red-500')
                }
              />
            </div>
          </CircularProgressbarWithChildren>
        </div>
      </div>
      {/* percentage removed for cleaner look */}
    </div>
  );
}
