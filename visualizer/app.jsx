"use strict";

const { useState, useEffect, useMemo, useRef } = React;

const DATA_FILE = "../visualizer/visualization_data.json";
const METRIC_LABELS = {
  elapsed: "Runtime (s)",
  cost: "Tour Cost",
};
const ERROR_LABELS = {
  ci95: "95% Confidence Interval",
  std: "Standard Deviation",
  range: "Range (min-max)",
};
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

const hexToRgba = (hex, alpha) => {
  const match = /^#?([a-f\d]{6})$/i.exec(hex);
  if (!match) {
    return `rgba(37, 99, 235, ${alpha})`;
  }
  const intVal = parseInt(match[1], 16);
  const r = (intVal >> 16) & 255;
  const g = (intVal >> 8) & 255;
  const b = intVal & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

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

function MetricControls({
  metrics,
  selectedMetrics,
  onToggleMetric,
  selectedErrorMetric,
  onErrorMetricChange,
  scaleType,
  onScaleChange,
}) {
  return (
    <section className="metric-toggle">
      <h3>Metrics</h3>
      <div className="metric-options">
        {metrics.map((metric) => {
          const id = `metric-${metric}`;
          const checked = selectedMetrics.includes(metric);
          const disableUncheck = checked && selectedMetrics.length === 1;
          const label = METRIC_LABELS[metric] ?? formatLabel(metric);
          return (
            <label key={id} className="metric-option" htmlFor={id}>
              <input
                id={id}
                type="checkbox"
                checked={checked}
                disabled={disableUncheck}
                onChange={() => onToggleMetric(metric)}
              />
              <span>{label}</span>
            </label>
          );
        })}
      </div>
      <p className="metric-hint">Select at least one metric to display.</p>

      <div className="radio-group" role="radiogroup" aria-label="Error metric">
        <div className="radio-option">
          <input
            type="radio"
            id="error-ci"
            value="ci95"
            name="error-metric"
            checked={selectedErrorMetric === "ci95"}
            onChange={(event) => onErrorMetricChange(event.target.value)}
          />
          <label htmlFor="error-ci">95% Confidence Interval</label>
        </div>
        <div className="radio-option">
          <input
            type="radio"
            id="error-std"
            value="std"
            name="error-metric"
            checked={selectedErrorMetric === "std"}
            onChange={(event) => onErrorMetricChange(event.target.value)}
          />
          <label htmlFor="error-std">Standard Deviation</label>
        </div>
        <div className="radio-option">
          <input
            type="radio"
            id="error-range"
            value="range"
            name="error-metric"
            checked={selectedErrorMetric === "range"}
            onChange={(event) => onErrorMetricChange(event.target.value)}
          />
          <label htmlFor="error-range">Range (min-max)</label>
        </div>
      </div>

      <div className="radio-group" role="radiogroup" aria-label="Axis scaling">
        <div className="radio-option">
          <input
            type="radio"
            id="scale-log"
            value="log"
            name="scale-type"
            checked={scaleType === "log"}
            onChange={(event) => onScaleChange(event.target.value)}
          />
          <label htmlFor="scale-log">Logarithmic axes</label>
        </div>
        <div className="radio-option">
          <input
            type="radio"
            id="scale-linear"
            value="linear"
            name="scale-type"
            checked={scaleType === "linear"}
            onChange={(event) => onScaleChange(event.target.value)}
          />
          <label htmlFor="scale-linear">Linear axes</label>
        </div>
      </div>
    </section>
  );
}

function App() {
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [selectedMetrics, setSelectedMetrics] = useState([]);
  const [selectedErrorMetric, setSelectedErrorMetric] = useState("ci95");
  const [selectedProblemTypes, setSelectedProblemTypes] = useState([]);
  const [selectedCategories, setSelectedCategories] = useState([]);
  const [selectedAlgorithms, setSelectedAlgorithms] = useState([]);
  const [selectedCityBuckets, setSelectedCityBuckets] = useState([]);
  const [scaleType, setScaleType] = useState("log");

  const chartRefs = useRef(new Set());

  useEffect(() => {
    async function loadData() {
      try {
        const response = await fetch(DATA_FILE, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Failed to load data (${response.status})`);
        }
        const data = await response.json();
        setPayload(data);

        const availableMetrics = data.metadata?.metrics ?? [];
        setSelectedMetrics(availableMetrics.length > 0 ? availableMetrics : ["elapsed"]);

        const problemOptions = data.metadata?.problem_types ?? ["all"];
        setSelectedProblemTypes(problemOptions.length > 0 ? problemOptions : ["all"]);

        setSelectedCategories(data.metadata?.algorithm_categories ?? []);
        setSelectedAlgorithms(data.metadata?.algorithms ?? []);
        setSelectedCityBuckets(data.metadata?.city_buckets ?? []);
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

  const toggleMetric = (metric) => {
    setSelectedMetrics((prev) => {
      if (prev.includes(metric)) {
        if (prev.length === 1) {
          return prev;
        }
        return prev.filter((item) => item !== metric);
      }
      return [...prev, metric];
    });
  };

  const bucketOrder = useMemo(() => {
    const order = new Map();
    (payload?.metadata?.city_buckets ?? []).forEach((bucket, idx) => order.set(bucket, idx));
    return order;
  }, [payload]);

  const filteredRecordsByMetric = useMemo(() => {
    if (!payload) {
      return new Map();
    }

    const activeMetrics =
      selectedMetrics.length > 0 ? selectedMetrics : payload.metadata?.metrics ?? [];
    const result = new Map(activeMetrics.map((metric) => [metric, []]));

    const useProblemFilter = selectedProblemTypes.length > 0;
    const useCategoryFilter = selectedCategories.length > 0;
    const useAlgorithmFilter = selectedAlgorithms.length > 0;
    const useCityFilter = selectedCityBuckets.length > 0;

    const problemSet = new Set(selectedProblemTypes);
    const categorySet = new Set(selectedCategories);
    const algorithmSet = new Set(selectedAlgorithms);
    const citySet = new Set(selectedCityBuckets);

    for (const record of payload.records) {
      if (!result.has(record.metric)) {
        continue;
      }
      const matchesProblem = !useProblemFilter || problemSet.has(record.problem_type);
      const matchesCategory = !useCategoryFilter || categorySet.has(record.algorithm_category);
      const matchesAlgorithm = !useAlgorithmFilter || algorithmSet.has(record.algorithm);
      const matchesCity = !useCityFilter || citySet.has(record.num_cities_bucket);
      if (matchesProblem && matchesCategory && matchesAlgorithm && matchesCity) {
        result.get(record.metric).push(record);
      }
    }

    return result;
  }, [payload, selectedMetrics, selectedProblemTypes, selectedCategories, selectedAlgorithms, selectedCityBuckets]);

  const chartDataByMetric = useMemo(() => {
    const errorKey = selectedErrorMetric === "ci95" ? "ci95" : "std";
    const charts = new Map();

    filteredRecordsByMetric.forEach((records, metric) => {
      const grouped = new Map();
      const ticks = new Set();

      for (const record of records) {
        if (!grouped.has(record.algorithm)) {
          grouped.set(record.algorithm, []);
        }
        grouped.get(record.algorithm).push(record);
      }

      const traces = [];

      for (const [algorithm, items] of grouped.entries()) {
        items.sort((a, b) => {
          const av = bucketOrder.get(a.num_cities_bucket) ?? Number.MAX_SAFE_INTEGER;
          const bv = bucketOrder.get(b.num_cities_bucket) ?? Number.MAX_SAFE_INTEGER;
          return av - bv;
        });

        const x = [];
        const center = [];
        const lower = [];
        const upper = [];
        const customdata = [];

        for (const item of items) {
          const centralValue = Math.max(item.mean ?? item.median ?? EPSILON, EPSILON);
          let lowerValue;
          let upperValue;

          if (selectedErrorMetric === "range") {
            const minValue = Number.isFinite(item.range_min) ? item.range_min : centralValue;
            const maxValue = Number.isFinite(item.range_max) ? item.range_max : centralValue;
            lowerValue = Math.max(minValue, EPSILON);
            upperValue = Math.max(maxValue, EPSILON);
          } else {
            const errorValue = Number.isFinite(item[errorKey]) ? Math.max(item[errorKey], 0) : 0;
            lowerValue = Math.max(centralValue - errorValue, EPSILON);
            upperValue = Math.max(centralValue + errorValue, EPSILON);
          }

          x.push(item.num_cities_bucket);
          center.push(centralValue);
          lower.push(lowerValue);
          upper.push(upperValue);
          customdata.push([
            item.mean ?? 0,
            item.median ?? 0,
            item.std ?? 0,
            item.ci95 ?? 0,
            item.count ?? 0,
            item.range_min ?? 0,
            item.range_max ?? 0,
          ]);
          ticks.add(item.num_cities_bucket);
        }

        const color = colorForAlgorithm(algorithm);
        const fillColor = hexToRgba(color, 0.18);
        const prettyName = formatLabel(algorithm);

        const fillX = [...x, ...x.slice().reverse()];
        const fillY = [...upper, ...lower.slice().reverse()];

        traces.push({
          x: fillX,
          y: fillY,
          mode: "lines",
          type: "scatter",
          showlegend: false,
          hoverinfo: "skip",
          legendgroup: algorithm,
          line: { width: 0 },
          fill: "toself",
          fillcolor: fillColor,
        });

        traces.push({
          x,
          y: center,
          mode: "lines+markers",
          type: "scatter",
          name: prettyName,
          legendgroup: algorithm,
          marker: { size: 7, color, opacity: 0.9, line: { width: 0 } },
          line: { width: 2, color, opacity: 0.85 },
          customdata,
          hovertemplate:
            "City bucket: %{x}<br>" +
            "Mean: %{customdata[0]:.3f}<br>" +
            "Median: %{customdata[1]:.3f}<br>" +
            "Std: %{customdata[2]:.3f}<br>" +
            "95% CI: ±%{customdata[3]:.3f}<br>" +
            "Range: %{customdata[5]:.3f} - %{customdata[6]:.3f}<br>" +
            "Runs: %{customdata[4]}<extra>" +
            prettyName +
            "</extra>",
        });
      }

      const tickValues = Array.from(ticks).sort((a, b) => {
        const av = bucketOrder.get(a) ?? Number.MAX_SAFE_INTEGER;
        const bv = bucketOrder.get(b) ?? Number.MAX_SAFE_INTEGER;
        return av - bv;
      });
      const metricLabel = METRIC_LABELS[metric] ?? formatLabel(metric);
      const layout = {
        template: "plotly_white",
        margin: { l: 70, r: 30, t: 50, b: 60 },
        xaxis: {
          title: "Number of Cities (bucketed)",
          type: "category",
          tickvals: tickValues,
          ticktext: tickValues,
        },
        yaxis: {
          title: metricLabel,
          type: scaleType,
        },
        legend: {
          orientation: "h",
          x: 0,
          y: 1.02,
          xanchor: "left",
          yanchor: "bottom",
        },
        hovermode: "closest",
        height: 520,
      };

      if (traces.length === 0) {
        layout.annotations = [
          {
            text: "No data matches the current filters.",
            x: 0.5,
            y: 0.5,
            xref: "paper",
            yref: "paper",
            showarrow: false,
            font: { size: 16, color: "#6b7280" },
          },
        ];
      }

      charts.set(metric, { traces, layout });
    });

    return charts;
  }, [filteredRecordsByMetric, selectedErrorMetric, scaleType]);

  useEffect(() => {
    const currentMetrics = new Set(chartDataByMetric ? chartDataByMetric.keys() : []);
    const previousMetrics = chartRefs.current;

    previousMetrics.forEach((metric) => {
      if (!currentMetrics.has(metric)) {
        const container = document.getElementById(`chart-${metric}`);
        if (container) {
          Plotly.purge(container);
        }
      }
    });

    chartDataByMetric.forEach((chart, metric) => {
      const container = document.getElementById(`chart-${metric}`);
      if (!container) {
        return;
      }
      Plotly.react(container, chart.traces, chart.layout, {
        responsive: true,
        displaylogo: false,
      });
    });

    chartRefs.current = currentMetrics;
  }, [chartDataByMetric]);

  const visibleAlgorithms = useMemo(() => {
    const set = new Set();
    filteredRecordsByMetric.forEach((records) => {
      records.forEach((record) => set.add(record.algorithm));
    });
    return set.size;
  }, [filteredRecordsByMetric]);

  const totalRuns = useMemo(() => {
    let sum = 0;
    filteredRecordsByMetric.forEach((records) => {
      records.forEach((record) => {
        sum += record.count ?? 0;
      });
    });
    return sum;
  }, [filteredRecordsByMetric]);

  return (
    <div className="app">
      <header>
        <h1>Interactive TSP Benchmark Visualiser</h1>
        <p className="description">
          Explore runtime and cost trends across algorithms, families, and problem types. Export data
          with <code>scripts/export_visualizer_data.py</code> and reload to refresh.
        </p>
        <nav className="nav-links">
          <a href="pareto.html">View Pareto Frontiers</a>
        </nav>
      </header>

      {loading && <p className="status">Loading aggregated data…</p>}
      {error && <div className="error">{error}</div>}

      {!loading && !error && payload && (
        <>
          <div className="panels">
            <div className="panel">
              <MultiSelectGroup
                label="Problem Types"
                options={payload.metadata.problem_types}
                selected={selectedProblemTypes}
                onChange={setSelectedProblemTypes}
              />
            </div>

            <div className="panel">
              <MultiSelectGroup
                label="Algorithm Families"
                options={payload.metadata.algorithm_categories}
                selected={selectedCategories}
                onChange={setSelectedCategories}
              />
            </div>

          <div className="panel">
            <MultiSelectGroup
              label="Algorithms"
              options={payload.metadata.algorithms}
              selected={selectedAlgorithms}
              onChange={setSelectedAlgorithms}
            />
          </div>

          <div className="panel">
            <MultiSelectGroup
              label="City Buckets"
              options={payload.metadata.city_buckets ?? []}
              selected={selectedCityBuckets}
              onChange={setSelectedCityBuckets}
              emptyMessage="No city buckets available."
            />
          </div>

            <div className="panel">
              <h2>Display Options</h2>
              <MetricControls
                metrics={payload.metadata.metrics}
                selectedMetrics={selectedMetrics}
                onToggleMetric={toggleMetric}
                selectedErrorMetric={selectedErrorMetric}
                onErrorMetricChange={setSelectedErrorMetric}
                scaleType={scaleType}
                onScaleChange={setScaleType}
              />
            </div>
          </div>

          <div className="chart-stack">
            {selectedMetrics.map((metric) => {
              const label = METRIC_LABELS[metric] ?? formatLabel(metric);
              const subtitle = ERROR_LABELS[selectedErrorMetric];
              return (
                <div className="chart-panel" key={metric}>
                  <h2>{label}</h2>
                  <p className="chart-subtitle">Shaded {subtitle}</p>
                  <div
                    id={`chart-${metric}`}
                    className="chart"
                    role="img"
                    aria-label={`Benchmark line chart for ${label}`}
                  />
                </div>
              );
            })}
          </div>

          <p className="status">
            Showing {visibleAlgorithms} algorithm{visibleAlgorithms === 1 ? "" : "s"} · Aggregated
            runs: {totalRuns}
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
