"use strict";

const { useState, useEffect, useMemo } = React;

const DATA_FILE = "../visualizer/visualization_data.json";
const COLOR_PALETTE = [
  "#2563eb",
  "#16a34a",
  "#ef4444",
  "#8b5cf6",
  "#f97316",
  "#0ea5e9",
  "#f59e0b",
  "#10b981",
  "#ec4899",
  "#6366f1",
  "#14b8a6",
];
const algorithmColorMap = new Map();
const EPSILON = 1e-9;

const formatLabel = (value) =>
  value
    .toString()
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

const slugify = (value) =>
  value
    .toString()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

const colorForAlgorithm = (algorithm) => {
  if (!algorithmColorMap.has(algorithm)) {
    const index = algorithmColorMap.size % COLOR_PALETTE.length;
    algorithmColorMap.set(algorithm, COLOR_PALETTE[index]);
  }
  return algorithmColorMap.get(algorithm);
};

function MultiSelectGroup({ label, options, selected, onChange, emptyMessage = "No options available." }) {
  const toggle = (option) => {
    if (selected.includes(option)) {
      onChange(selected.filter((value) => value !== option));
    } else {
      onChange([...selected, option]);
    }
  };

  const selectAll = () => onChange([...options]);
  const clearAll = () => onChange([]);
  const allSelected = selected.length === options.length && options.length > 0;

  return (
    <section className="filter-group">
      <div className="filter-heading">
        <h3>{label}</h3>
        {options.length > 0 && (
          <button type="button" onClick={allSelected ? clearAll : selectAll}>
            {allSelected ? "Clear" : "Select all"}
          </button>
        )}
      </div>
      <div className="filter-options">
        {options.length === 0 && <span>{emptyMessage}</span>}
        {options.map((option) => {
          const id = `${slugify(label)}-${slugify(option)}`;
          return (
            <label key={id} className="filter-option" htmlFor={id}>
              <input
                id={id}
                type="checkbox"
                checked={selected.includes(option)}
                onChange={() => toggle(option)}
              />
              <span>{formatLabel(option)}</span>
            </label>
          );
        })}
      </div>
    </section>
  );
}

function AxisControls({ xScale, yScale, onXScaleChange, onYScaleChange }) {
  return (
    <section className="metric-toggle">
      <h3>Axis Scaling</h3>
      <div className="radio-group" role="radiogroup" aria-label="Cost axis scaling">
        <div className="radio-option">
          <input
            type="radio"
            id="x-log"
            value="log"
            name="x-scale"
            checked={xScale === "log"}
            onChange={(event) => onXScaleChange(event.target.value)}
          />
          <label htmlFor="x-log">Cost: log scale</label>
        </div>
        <div className="radio-option">
          <input
            type="radio"
            id="x-linear"
            value="linear"
            name="x-scale"
            checked={xScale === "linear"}
            onChange={(event) => onXScaleChange(event.target.value)}
          />
          <label htmlFor="x-linear">Cost: linear scale</label>
        </div>
      </div>

      <div className="radio-group" role="radiogroup" aria-label="Runtime axis scaling">
        <div className="radio-option">
          <input
            type="radio"
            id="y-log"
            value="log"
            name="y-scale"
            checked={yScale === "log"}
            onChange={(event) => onYScaleChange(event.target.value)}
          />
          <label htmlFor="y-log">Runtime: log scale</label>
        </div>
        <div className="radio-option">
          <input
            type="radio"
            id="y-linear"
            value="linear"
            name="y-scale"
            checked={yScale === "linear"}
            onChange={(event) => onYScaleChange(event.target.value)}
          />
          <label htmlFor="y-linear">Runtime: linear scale</label>
        </div>
      </div>
    </section>
  );
}

const computeParetoFrontier = (records) => {
  if (!records || records.length === 0) {
    return [];
  }
  const points = records
    .map((record) => {
      const runtime = Number(record.runtime_mean);
      const cost = Number(record.cost_mean);
      if (!Number.isFinite(runtime) || !Number.isFinite(cost)) {
        return null;
      }
      return {
        record,
        runtime: Math.max(runtime, EPSILON),
        cost: Math.max(cost, EPSILON),
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.cost - b.cost);

  const frontier = [];
  let bestRuntime = Infinity;
  for (const point of points) {
    if (point.runtime <= bestRuntime + EPSILON) {
      frontier.push(point);
      bestRuntime = Math.min(bestRuntime, point.runtime);
    }
  }
  return frontier;
};

function App() {
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [selectedCategories, setSelectedCategories] = useState([]);
  const [selectedAlgorithms, setSelectedAlgorithms] = useState([]);
  const [selectedCityBuckets, setSelectedCityBuckets] = useState([]);
  const [selectedProblemTypes, setSelectedProblemTypes] = useState([]);
  const [xScale, setXScale] = useState("log");
  const [yScale, setYScale] = useState("log");
  const bucketOrder = useMemo(() => {
    const order = new Map();
    (payload?.metadata?.city_buckets ?? []).forEach((b, idx) => order.set(b, idx));
    return order;
  }, [payload]);

  useEffect(() => {
    async function loadData() {
      try {
        const response = await fetch(DATA_FILE, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Failed to load data (${response.status})`);
        }
        const data = await response.json();
        setPayload(data);
        setSelectedCategories(data.metadata?.algorithm_categories ?? []);
        setSelectedAlgorithms(data.metadata?.algorithms ?? []);
        setSelectedCityBuckets(data.metadata?.city_buckets ?? []);
        setSelectedProblemTypes(data.metadata?.problem_types ?? ["all"]);
      } catch (err) {
        console.error(err);
        setError(
          "Unable to load aggregated results. Generate them with `python visualizer/results_export_for_visualizer.py`."
        );
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  const algorithmFamilies = useMemo(
    () => payload?.metadata?.algorithm_categories ?? [],
    [payload]
  );
  const allAlgorithms = useMemo(() => payload?.metadata?.algorithms ?? [], [payload]);
  const allCityBuckets = useMemo(() => payload?.metadata?.city_buckets ?? [], [payload]);
  const allProblemTypes = useMemo(
    () => payload?.metadata?.problem_types ?? ["all"],
    [payload]
  );
  const supportsRuntimeCost = payload?.metadata?.supports_runtime_cost ?? false;

  const filteredFrontiers = useMemo(() => {
    if (!payload || !supportsRuntimeCost) {
      return new Map();
    }

    const categorySet =
      selectedCategories.length > 0 ? new Set(selectedCategories) : null;
    const algorithmSet =
      selectedAlgorithms.length > 0 ? new Set(selectedAlgorithms) : null;
    const cityBucketSet =
      selectedCityBuckets.length > 0 ? new Set(selectedCityBuckets) : null;
    const problemSet =
      selectedProblemTypes.length > 0 ? new Set(selectedProblemTypes) : null;

    const grouped = new Map();

    for (const record of payload.records) {
      if (record.metric !== "runtime_cost") {
        continue;
      }
      if (categorySet && !categorySet.has(record.algorithm_category)) {
        continue;
      }
      if (algorithmSet && !algorithmSet.has(record.algorithm)) {
        continue;
      }
      if (problemSet && !problemSet.has(record.problem_type)) {
        continue;
      }
      if (cityBucketSet && !cityBucketSet.has(record.num_cities_bucket)) {
        continue;
      }
      if (!grouped.has(record.algorithm)) {
        grouped.set(record.algorithm, []);
      }
      grouped.get(record.algorithm).push(record);
    }

    const result = new Map();
    grouped.forEach((records, algorithm) => {
      const frontier = computeParetoFrontier(records);
      if (frontier.length > 0) {
        result.set(algorithm, frontier);
      }
    });

    return result;
  }, [payload, supportsRuntimeCost, selectedCategories, selectedAlgorithms, selectedCityBuckets]);

  const chartData = useMemo(() => {
    if (!supportsRuntimeCost) {
      return { traces: [], layout: {} };
    }
    const traces = [];
    filteredFrontiers.forEach((frontier, algorithm) => {
      const color = colorForAlgorithm(algorithm);
      const prettyName = formatLabel(algorithm);
      // Sort frontier points by bucket order then cost for nicer hovers/lines
      frontier.sort((a, b) => {
        const av = bucketOrder.get(a.record.num_cities_bucket) ?? Number.MAX_SAFE_INTEGER;
        const bv = bucketOrder.get(b.record.num_cities_bucket) ?? Number.MAX_SAFE_INTEGER;
        if (av !== bv) return av - bv;
        return (a.record.cost_mean ?? a.cost) - (b.record.cost_mean ?? b.cost);
      });
      const x = frontier.map((point) => point.cost);
      const y = frontier.map((point) => point.runtime);
      const hoverText = frontier.map(({ record, runtime, cost }) => {
        const bucket = record.num_cities_bucket ?? "unknown";
        const rMin = record.runtime_min ?? runtime;
        const rMax = record.runtime_max ?? runtime;
        const cMin = record.cost_min ?? cost;
        const cMax = record.cost_max ?? cost;
        const runs = record.count ?? 0;
        return [
          `City bucket: ${bucket}`,
          `Mean runtime: ${runtime.toFixed(4)}s`,
          `Runtime range: ${rMin.toFixed(4)}s - ${rMax.toFixed(4)}s`,
          `Mean tour cost: ${cost.toFixed(2)}`,
          `Cost range: ${cMin.toFixed(2)} - ${cMax.toFixed(2)}`,
          `Runs: ${runs}`,
        ].join("<br>");
      });
      traces.push({
        x,
        y,
        mode: "lines+markers",
        type: "scatter",
        name: prettyName,
        line: { width: 2, color, opacity: 0.85 },
        marker: { size: 8, color, opacity: 0.95, line: { width: 0 } },
        text: hoverText,
        hoverinfo: "text",
        hovertemplate: "%{text}<extra>" + prettyName + "</extra>",
      });
    });

    const layout = {
      template: "plotly_white",
      margin: { l: 70, r: 30, t: 50, b: 60 },
      xaxis: {
        title: "Mean Tour Cost",
        type: xScale,
      },
      yaxis: {
        title: "Mean Runtime (s)",
        type: yScale,
      },
      legend: {
        orientation: "h",
        x: 0,
        y: 1.05,
        xanchor: "left",
        yanchor: "bottom",
      },
      hovermode: "closest",
      height: 600,
    };

    if (traces.length === 0) {
      layout.annotations = [
        {
          text: "No algorithms match the current filters.",
          x: 0.5,
          y: 0.5,
          xref: "paper",
          yref: "paper",
          showarrow: false,
          font: { size: 16, color: "#6b7280" },
        },
      ];
    }

    return { traces, layout };
  }, [filteredFrontiers, xScale, yScale, supportsRuntimeCost]);

  useEffect(() => {
    if (!chartData || !supportsRuntimeCost) {
      return;
    }
    const container = document.getElementById("pareto-chart");
    if (!container) {
      return;
    }
    Plotly.react(container, chartData.traces, chartData.layout, {
      responsive: true,
      displaylogo: false,
    });
  }, [chartData, supportsRuntimeCost]);

  const visibleAlgorithms = useMemo(() => filteredFrontiers.size, [filteredFrontiers]);
  const paretoPoints = useMemo(() => {
    let count = 0;
    filteredFrontiers.forEach((frontier) => {
      count += frontier.length;
    });
    return count;
  }, [filteredFrontiers]);

  return (
    <div className="app">
      <header>
        <h1>TSP Pareto Frontiers</h1>
        <p className="description">
          Compare the trade-off between mean runtime and mean tour cost for each algorithm across
          selected city counts. Points are non-dominated with respect to minimising both objectives.
        </p>
        <nav className="nav-links">
          <a href="index.html">Back to Benchmark Visualiser</a>
        </nav>
      </header>

      {loading && <p className="status">Loading aggregated data…</p>}
      {error && <div className="error">{error}</div>}
      {!loading && !error && payload && !supportsRuntimeCost && (
        <div className="error">
          Runtime/cost aggregates are unavailable. Re-export results with{" "}
          <code>scripts/export_visualizer_data.py</code> to populate this view.
        </div>
      )}

      {!loading && !error && payload && supportsRuntimeCost && (
        <>
          <div className="panels">
            <div className="panel">
              <MultiSelectGroup
                label="Algorithm Families"
                options={algorithmFamilies}
                selected={selectedCategories}
                onChange={setSelectedCategories}
              />
            </div>
            <div className="panel">
              <MultiSelectGroup
                label="Problem Types"
                options={allProblemTypes}
                selected={selectedProblemTypes}
                onChange={setSelectedProblemTypes}
                emptyMessage="No problem types available."
              />
            </div>
            <div className="panel">
              <MultiSelectGroup
                label="City Buckets"
                options={allCityBuckets}
                selected={selectedCityBuckets}
                onChange={setSelectedCityBuckets}
                emptyMessage="No city buckets available."
              />
            </div>
            <div className="panel">
              <MultiSelectGroup
                label="Algorithms"
                options={allAlgorithms}
                selected={selectedAlgorithms}
                onChange={setSelectedAlgorithms}
              />
            </div>
            <div className="panel">
              <h2>Display Options</h2>
              <AxisControls
                xScale={xScale}
                yScale={yScale}
                onXScaleChange={setXScale}
                onYScaleChange={setYScale}
              />
            </div>
          </div>

          <div className="chart-panel">
            <div
              id="pareto-chart"
              className="chart"
              role="img"
              aria-label="Pareto frontier chart for runtime versus city count"
            />
          </div>

          <p className="status">
            Showing {visibleAlgorithms} algorithm{visibleAlgorithms === 1 ? "" : "s"} with{" "}
            {paretoPoints} Pareto point{paretoPoints === 1 ? "" : "s"}.
          </p>
          <p className="status">
            Data generated from {payload.metadata.source_results} on{" "}
            {new Date(payload.metadata.generated_at).toLocaleString()}
          </p>
        </>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
