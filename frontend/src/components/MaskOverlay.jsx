import { useState } from "react";
import { priorityTone, severityTone, TONE_COLORS } from "../scoring";

/**
 * A photo with each detection's mask_polygon drawn on top. The SVG viewBox is the
 * image's natural size, so polygon coords (original-image pixels) line up at any
 * display size. `image-orientation: none` (in CSS) keeps the displayed pixels in the
 * same orientation the backend analysed.
 *
 * detections: [{ id, mask_polygon, label, category, kind }]
 */
export default function MaskOverlay({ src, alt, detections = [], activeId, onSelect, fallbackSize }) {
  const [natural, setNatural] = useState(null);
  const size = natural || fallbackSize;

  return (
    <figure className="mask-figure">
      <img
        src={src}
        alt={alt}
        className="mask-img"
        onLoad={(e) => setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
      />
      {size && size.w > 0 && (
        <svg className="mask-svg" viewBox={`0 0 ${size.w} ${size.h}`} preserveAspectRatio="none" aria-hidden="true">
          {detections.map((d) => {
            const pts = Array.isArray(d.mask_polygon) ? d.mask_polygon : [];
            if (pts.length < 3) return null;
            const tone = d.kind === "priority" ? priorityTone(d.category) : severityTone(d.category);
            const color = TONE_COLORS[tone];
            const anchor = topmost(pts);
            const unit = Math.max(size.w, size.h) / 100; // label sizing relative to the photo
            const active = activeId == null || activeId === d.id;
            return (
              <g
                key={d.id}
                className={`mask-shape${active ? " is-active" : ""}`}
                onClick={onSelect ? () => onSelect(d.id) : undefined}
              >
                <polygon
                  points={pts.map(([x, y]) => `${x},${y}`).join(" ")}
                  fill={color}
                  fillOpacity={active ? 0.28 : 0.1}
                  stroke="#fff"
                  strokeWidth={unit * 0.9}
                  strokeLinejoin="round"
                />
                <polygon
                  points={pts.map(([x, y]) => `${x},${y}`).join(" ")}
                  fill="none"
                  stroke={color}
                  strokeWidth={unit * 0.45}
                  strokeLinejoin="round"
                  strokeOpacity={active ? 1 : 0.5}
                />
                {d.label != null && (
                  <g transform={`translate(${anchor[0]} ${Math.max(anchor[1] - unit * 3.2, unit * 3)})`}>
                    <circle r={unit * 2.8} fill={color} stroke="#fff" strokeWidth={unit * 0.6} />
                    <text
                      textAnchor="middle"
                      dominantBaseline="central"
                      fontSize={unit * 3.2}
                      fontWeight="700"
                      fill="#fff"
                      fontFamily="Inter, system-ui, sans-serif"
                    >
                      {d.label}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </svg>
      )}
    </figure>
  );
}

function topmost(points) {
  let best = points[0];
  for (const p of points) if (p[1] < best[1]) best = p;
  return best;
}
