import React, { useState } from "react";

const NULL_DELTAS_V2 = [-0.59375,-0.375,-0.0625,-0.03125,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.03125,0.03125,0.03125,0.0625,0.09375,0.28125];
const PAPER_DELTA_V2 = -0.03125;

// v1 data for comparison
const PAPER_DELTA_V1 = -2.609375;
const ROUNDTRIP_V1 = -1.875;

export default function NullDistributionV2() {
  const [showV1] = useState(true);
  const [hoveredBar, setHoveredBar] = useState(null);

  // Chart dimensions
  const W = 780, H = 420;
  const margin = { top: 60, right: 30, bottom: 100, left: 70 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  // Bin the data — use narrow bins to show the zero spike clearly
  const binWidth = 0.1;
  const minVal = -0.7;
  const maxVal = 0.4;
  const bins = [];
  for (let edge = minVal; edge < maxVal; edge += binWidth) {
    const lo = edge;
    const hi = edge + binWidth;
    const count = NULL_DELTAS_V2.filter(d => d >= lo && d < hi).length;
    bins.push({ lo, hi, count, mid: (lo + hi) / 2 });
  }
  // Last bin includes upper edge
  bins[bins.length - 1].count += NULL_DELTAS_V2.filter(d => d >= maxVal).length;

  const maxCount = Math.max(...bins.map(b => b.count));
  
  // Scales
  const xScale = (val) => margin.left + ((val - minVal) / (maxVal - minVal)) * plotW;
  const yScale = (count) => margin.top + plotH - (count / maxCount) * plotH;
  const barW = (binWidth / (maxVal - minVal)) * plotW;

  // Axis ticks
  const xTicks = [];
  for (let v = minVal; v <= maxVal + 0.01; v += 0.1) {
    xTicks.push(Math.round(v * 100) / 100);
  }
  const yTicks = [0, 20, 40, 60, 80, 100, 120, 140];

  return (
    <div style={{ background: "#0d1117", padding: "24px", borderRadius: "12px", fontFamily: "'Inter', system-ui, sans-serif", maxWidth: "840px" }}>
      <div style={{ color: "#e6edf3", fontSize: "16px", fontWeight: 600, marginBottom: "4px" }}>
        EXP H v2: Null Distribution (Delta-Steering, Clean)
      </div>
      <div style={{ color: "#8b949e", fontSize: "12px", marginBottom: "16px" }}>
        150 random 2-feature combos vs paper features [1795, 934] · Gemma 3 4B · Layer 22 · Hard ablation
      </div>

      <svg width={W} height={H} style={{ display: "block" }}>
        {/* Grid lines */}
        {yTicks.map(t => (
          <line key={t} x1={margin.left} x2={W - margin.right}
            y1={yScale(t)} y2={yScale(t)}
            stroke="#21262d" strokeWidth={1} />
        ))}

        {/* Bars */}
        {bins.map((bin, i) => {
          if (bin.count === 0) return null;
          const x = xScale(bin.lo) + 1;
          const w = barW - 2;
          const y = yScale(bin.count);
          const h = plotH + margin.top - y;
          const isZeroBin = bin.lo <= 0 && bin.hi > 0;
          const isPaperBin = bin.lo <= PAPER_DELTA_V2 && bin.hi > PAPER_DELTA_V2;
          
          return (
            <g key={i}
              onMouseEnter={() => setHoveredBar(bin)}
              onMouseLeave={() => setHoveredBar(null)}>
              <rect x={x} y={y} width={w} height={h}
                fill={isZeroBin ? "#1f6feb" : "#388bfd"}
                opacity={hoveredBar === bin ? 1 : 0.8}
                rx={2} />
              {bin.count > 2 && (
                <text x={x + w/2} y={y - 5} textAnchor="middle"
                  fill="#8b949e" fontSize="11">{bin.count}</text>
              )}
            </g>
          );
        })}

        {/* Paper features marker */}
        <line x1={xScale(PAPER_DELTA_V2)} x2={xScale(PAPER_DELTA_V2)}
          y1={margin.top - 10} y2={margin.top + plotH}
          stroke="#f85149" strokeWidth={2.5} strokeDasharray="6,3" />
        <text x={xScale(PAPER_DELTA_V2)} y={margin.top - 18}
          textAnchor="middle" fill="#f85149" fontSize="11" fontWeight="600">
          Paper: −0.031
        </text>

        {/* v1 comparison (off-chart left, with arrow) */}
        {showV1 && (
          <g>
            <line x1={margin.left} x2={margin.left + 18}
              y1={margin.top + plotH * 0.35} y2={margin.top + plotH * 0.35}
              stroke="#f0883e" strokeWidth={2} markerEnd="url(#arrowLeft)" />
            <text x={margin.left + 22} y={margin.top + plotH * 0.35 - 8}
              fill="#f0883e" fontSize="10" fontWeight="500">
              v1: −2.609
            </text>
            <text x={margin.left + 22} y={margin.top + plotH * 0.35 + 6}
              fill="#f0883e" fontSize="9" opacity={0.7}>
              (off chart — 98.8% was artifact)
            </text>
          </g>
        )}

        {/* Arrow marker definition */}
        <defs>
          <marker id="arrowLeft" markerWidth="8" markerHeight="6"
            refX="8" refY="3" orient="auto">
            <path d="M8,0 L0,3 L8,6" fill="#f0883e" />
          </marker>
        </defs>

        {/* X axis */}
        <line x1={margin.left} x2={W - margin.right}
          y1={margin.top + plotH} y2={margin.top + plotH}
          stroke="#30363d" strokeWidth={1} />
        {xTicks.map(v => (
          <g key={v}>
            <line x1={xScale(v)} x2={xScale(v)}
              y1={margin.top + plotH} y2={margin.top + plotH + 5}
              stroke="#30363d" />
            <text x={xScale(v)} y={margin.top + plotH + 18}
              textAnchor="middle" fill="#8b949e" fontSize="10">
              {v.toFixed(1)}
            </text>
          </g>
        ))}
        <text x={margin.left + plotW / 2} y={margin.top + plotH + 36}
          textAnchor="middle" fill="#8b949e" fontSize="12">
          Logit-difference delta (YES−NO shift from baseline)
        </text>

        {/* Y axis */}
        {yTicks.map(t => (
          <text key={t} x={margin.left - 8} y={yScale(t) + 4}
            textAnchor="end" fill="#8b949e" fontSize="10">{t}</text>
        ))}
        <text x={margin.left - 45} y={margin.top + plotH / 2}
          textAnchor="middle" fill="#8b949e" fontSize="12"
          transform={`rotate(-90, ${margin.left - 45}, ${margin.top + plotH / 2})`}>
          Count (of 150)
        </text>

        {/* Tooltip */}
        {hoveredBar && (
          <g>
            <rect x={xScale(hoveredBar.mid) - 70} y={yScale(hoveredBar.count) - 42}
              width={140} height={32} rx={4} fill="#161b22" stroke="#30363d" />
            <text x={xScale(hoveredBar.mid)} y={yScale(hoveredBar.count) - 22}
              textAnchor="middle" fill="#e6edf3" fontSize="11">
              [{hoveredBar.lo.toFixed(2)}, {hoveredBar.hi.toFixed(2)}): {hoveredBar.count} pairs
            </text>
          </g>
        )}
      </svg>

      {/* Stats panel */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px", marginTop: "16px" }}>
        <div style={{ background: "#161b22", borderRadius: "8px", padding: "12px", border: "1px solid #21262d" }}>
          <div style={{ color: "#8b949e", fontSize: "11px", marginBottom: "4px" }}>Paper features (v2)</div>
          <div style={{ color: "#f85149", fontSize: "20px", fontWeight: 700 }}>−0.031</div>
          <div style={{ color: "#8b949e", fontSize: "10px" }}>cond. p = 0.40 (middle)</div>
        </div>
        <div style={{ background: "#161b22", borderRadius: "8px", padding: "12px", border: "1px solid #21262d" }}>
          <div style={{ color: "#8b949e", fontSize: "11px", marginBottom: "4px" }}>Null distribution</div>
          <div style={{ color: "#58a6ff", fontSize: "20px", fontWeight: 700 }}>140/150 = 0</div>
          <div style={{ color: "#8b949e", fontSize: "10px" }}>10 non-zero, range [−0.59, +0.28]</div>
        </div>
        <div style={{ background: "#161b22", borderRadius: "8px", padding: "12px", border: "1px solid #21262d" }}>
          <div style={{ color: "#8b949e", fontSize: "11px", marginBottom: "4px" }}>v1 → v2 change</div>
          <div style={{ color: "#f0883e", fontSize: "20px", fontWeight: 700 }}>98.8%</div>
          <div style={{ color: "#8b949e", fontSize: "10px" }}>of v1 effect was artifact</div>
        </div>
      </div>

      {/* Interpretation */}
      <div style={{ background: "#161b22", borderRadius: "8px", padding: "12px", marginTop: "12px", border: "1px solid #21262d" }}>
        <div style={{ color: "#e6edf3", fontSize: "12px", lineHeight: 1.5 }}>
          <strong style={{ color: "#f85149" }}>Interpretation:</strong> With the reconstruction confound removed, 
          "consciousness features" produce a delta indistinguishable from zero (−0.031 vs null mean −0.004). 
          The v1 result (−2.609) — 98.8% of it disappeared after switching to delta-steering. Among the 10 non-zero random pairs, 
          paper features rank 5th out of 11 — dead center. These features do nothing to consciousness-related outputs.
        </div>
      </div>
    </div>
  );
}
