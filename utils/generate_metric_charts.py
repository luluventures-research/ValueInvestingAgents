"""
generate_metric_charts.py
--------------------------------------------------
Create 1920×1080 Apple‑Keynote‑style line charts (English or Chinese)
for each metric in a 10‑year history table, with Buffett's standard lines where applicable.

Usage:
    python generate_metric_charts.py --input metrics.csv \
           --font SourceHanSansSC-Bold.otf \
           --output metrics \
           --theme dark \
           --language chinese

Input file format (CSV or Excel):
    Year,Metric1,Metric2,...
    2015,...,...
    2016,...,...

• The first column must be named "Year".
• For bilingual labels, name a column like:
      "Revenue ($B)|收入（十亿美元）"
  (English and Chinese separated by a pipe "|").
• Percent metrics should include a "%" in the header so the script auto‑formats the y‑axis.
• Ratio metrics can contain "P/" or "/" (e.g. "Debt/Equity").
• Buffett's standard lines are drawn for metrics like P/E, P/B, Debt/Equity, etc.

Produces:
    <input_filename>/1. <metric>.png  (for each metric, with sanitized filenames in chosen language)
    metric_charts.zip       (all images zipped)
--------------------------------------------------
"""

import argparse, os, gc, sys, pandas as pd, matplotlib.pyplot as plt
import matplotlib.ticker as mtick, matplotlib.font_manager as fm
from zipfile import ZipFile
import re

# ---------- Config (feel free to tweak) ----------
FIGSIZE = (19.2, 10.8)   # inches → 1920×1080 at dpi=100
TITLE_SIZE = 36
LABEL_SIZE = 28
TICK_SIZE = 24
LINE_WIDTH = 3
MARKER_SIZE = 8
ANNOTATION_OFFSET = 20  # Base offset in points for annotations
MIN_DISTANCE = 0.1      # Minimum y-value difference to consider annotations or lines separate
LINE_PROXIMITY_THRESHOLD = 0.5  # Minimum y-distance from lines to avoid overlap
MAX_RECURSION = 10      # Maximum recursion depth to prevent stack overflow

# Buffett's standard thresholds with direction
BUFFETT_STANDARDS = {
    'P/E': {'threshold': 15, 'direction': '≤'},  # Price to Earnings ratio ≤ 15
    'P/B': {'threshold': 1.5, 'direction': '≤'},  # Price to Book ratio ≤ 1.5
    'Debt/Equity': {'threshold': 0.5, 'direction': '<'},  # Debt-to-Equity < 0.5
    'Interest Coverage': {'threshold': 3, 'direction': '≥'},  # EBIT / Interest ≥ 3
    'Interest Expense/Operating Income': {'threshold': 0.15, 'direction': '≤'},  # ≤ 15%
    'Current Ratio': {'threshold': 1.5, 'direction': '>'},  # > 1.5
    'Gross Margin': {'threshold': 40, 'direction': '≳'},  # ≳ 40%
    'Net Profit Margin': {'threshold': 20, 'direction': '≳'},  # ≳ 20%
    'ROE': {'threshold': 15, 'direction': '≥'},  # Return on Equity ≥ 15%
    'ROIC': {'threshold': 12, 'direction': '≥'},  # Return on Invested Capital ≥ 12%
    'ROTC': {'threshold': 12, 'direction': '≥'},  # Return on Total Capital ≥ 12%
    'ROA': {'threshold': 6, 'direction': '≥'},  # Return on Assets ≥ 6%
    'Depreciation/Gross Profit': {'threshold': 0.1, 'direction': '≤'},  # ≤ 10%
    'FCF/Revenue': {'threshold': 5, 'direction': '>'},  # Free Cash Flow / Revenue > 5%
    'Return on Retained Earnings': {'threshold': 12, 'direction': '≥'},  # ≥ 12%
    'Owner’s-earnings/Market Cap': {'threshold': 10, 'direction': '≥'}  # ≥ 10%
}

# Theme configurations
THEMES = {
    'dark': {
        'figure.facecolor': '#1A1A1A',  # Deep charcoal
        'axes.facecolor': '#1A1A1A',
        'axes.edgecolor': '#E0E0E0',    # Soft white
        'axes.labelcolor': '#E0E0E0',
        'xtick.color': '#E0E0E0',
        'ytick.color': '#E0E0E0',
        'line_color': '#40C4FF',        # Bright cyan
        'avg_line_color': '#FFCA28',    # Warm amber
        'buffett_line_color': '#FF5252',# Bold red for Buffett's standard
        'latest_annot_color': '#26A69A', # Teal
        'max_annot_color': '#66BB6A',   # Light green
        'min_annot_color': '#EF5350',   # Soft red
        'title_color': '#E0E0E0',
        'spine_color': '#E0E0E0'
    },
    'light': {
        'figure.facecolor': '#F8FAFC',  # Clean off-white
        'axes.facecolor': '#F8FAFC',
        'axes.edgecolor': '#2D3748',    # Dark slate
        'axes.labelcolor': '#2D3748',
        'xtick.color': '#2D3748',
        'ytick.color': '#2D3748',
        'line_color': '#2563EB',        # Rich blue
        'avg_line_color': '#F59E0B',    # Vibrant amber
        'buffett_line_color': '#DC2626', # Deep red for Buffett's standard
        'latest_annot_color': '#9333EA', # Vivid purple
        'max_annot_color': '#16A34A',   # Bright green
        'min_annot_color': '#DB2777',   # Bold pink
        'title_color': '#2D3748',
        'spine_color': '#2D3748'
    }
}

# Language-specific labels for annotations and x-axis
LABELS = {
    'english': {
        'avg': 'Avg',
        'buffett': 'Buffett Standard',
        'latest': 'Latest',
        'max': 'Max',
        'min': 'Min',
        'xlabel': 'Fiscal Year'
    },
    'chinese': {
        'avg': '平均值',
        'buffett': '巴菲特标准',
        'latest': '最新',
        'max': '最高',
        'min': '最低',
        'xlabel': '财政年度'
    }
}

# Language-specific y-axis labels
YAXIS_LABELS = {
    'english': {
        '%': '%',
        'Ratio (x)': 'Ratio (x)',
        'USD': 'USD',
        'Price (USD)': 'Price (USD)',
        'USD Billions': 'USD Billions',
        'USD Millions': 'USD Millions',
        '': ''
    },
    'chinese': {
        '%': '%',
        'Ratio (x)': '比率 (x)',
        'USD': '美元',
        'Price (USD)': '价格 (美元)',
        'USD Billions': '亿美元',
        'USD Millions': '百万美元',
        '': ''
    }
}

# -------------------------------------------------
def sanitize_filename(filename: str) -> str:
    """Sanitize filename by replacing invalid characters with underscores."""
    invalid_chars = r'[\/:*?"<>|%]'
    sanitized = re.sub(invalid_chars, '_', filename)
    sanitized = sanitized.strip().replace('  ', ' ')
    return sanitized

def detect_ylabel(metric_name: str, language: str):
    """Detect y-axis label based on metric name and language."""
    rate_metrics = [
        'Revenue Growth Rate', 'Gross Margin', 'Operation Margin',
        'Net Income Growth Rate', 'Net Income Margin', 'Free Cash Flow Margin',
        'Owner\'s-Earnings Margin', 'Return on Retained Earnings',
        'Return On Invested Capital (ROIC)', 'Return on Equity (ROE)',
        'Return on Assets (ROA)', 'Owner\'s-Earnings/Total Market Cap',
        'Interest Expense / Operating Income', 'Depreciation / Gross Profit'
    ]
    if any(rate in metric_name for rate in rate_metrics) or '%' in metric_name:
        return YAXIS_LABELS[language]['%']
    if 'Ratio' in metric_name or '/' in metric_name or metric_name.startswith('P/'):
        return YAXIS_LABELS[language]['Ratio (x)']
    if '$' in metric_name or 'USD' in metric_name:
        return YAXIS_LABELS[language]['USD']
    if 'Price' in metric_name:
        return YAXIS_LABELS[language]['Price (USD)']
    if '($B)' in metric_name:
        return YAXIS_LABELS[language]['USD Billions']
    if '($M)' in metric_name:
        return YAXIS_LABELS[language]['USD Millions']
    return YAXIS_LABELS[language]['']

def get_buffett_standard(metric_name: str):
    """Check if metric matches Buffett's standards and return threshold and direction if applicable."""
    for key, info in BUFFETT_STANDARDS.items():
        if key.lower() in metric_name.lower().replace(' ', ''):
            return key, info['threshold'], info['direction']
    return None, None, None

def load_table(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    return pd.read_csv(path)

def adjust_annotation_position(ax, fig, x, y, text, color, offset_x=10, offset_y=0, avoid=None, line_y_positions=None, recursion_depth=0):
    """Adjust annotation position to avoid overlap with other annotations, lines, and stay within plot boundaries."""
    if recursion_depth > MAX_RECURSION:
        return ax.annotate(text, xy=(x, y), xytext=(offset_x, 0), textcoords='offset points', 
                           color=color, ha='right', va='center')  # Fallback with minimal offset

    annot = ax.annotate(text, xy=(x, y), xytext=(offset_x, offset_y), 
                        textcoords='offset points', color=color, ha='right', va='center')
    renderer = fig.canvas.get_renderer()
    bbox1 = annot.get_window_extent(renderer=renderer)
    y_limits = ax.get_ylim()
    data_y = y + offset_y / 72  # Approximate data y-position (assuming 72 dpi)

    # Check if annotation is within y-limits
    if bbox1.y0 < ax.get_window_extent().y0 or bbox1.y1 > ax.get_window_extent().y1:
        if data_y + offset_y / 72 > y_limits[1]:
            offset_y = int((y_limits[1] - y) * 72 - offset_y)  # Move down
        elif data_y + offset_y / 72 < y_limits[0]:
            offset_y = int((y_limits[0] - y) * 72 - offset_y)  # Move up
        annot.remove()
        return adjust_annotation_position(ax, fig, x, y, text, color, offset_x, offset_y, avoid, line_y_positions, recursion_depth + 1)

    # Check overlap with other annotations and enforce minimum separation
    if avoid:
        for other_annot in avoid:
            bbox2 = other_annot.get_window_extent(renderer=renderer)
            if bbox1.overlaps(bbox2) or abs(y - other_annot.get_position()[1]) < MIN_DISTANCE:
                # If annotations share the same y-value or overlap, add offset
                offset_y += ANNOTATION_OFFSET * (1 + avoid.index(other_annot))  # Incremental offset
                annot.remove()
                return adjust_annotation_position(ax, fig, x, y, text, color, offset_x, offset_y, avoid, line_y_positions, recursion_depth + 1)

    # Check overlap with lines
    if line_y_positions:
        for line_y in line_y_positions:
            if abs(data_y - line_y) < LINE_PROXIMITY_THRESHOLD:
                offset_y += ANNOTATION_OFFSET
                annot.remove()
                return adjust_annotation_position(ax, fig, x, y, text, color, offset_x, offset_y, avoid, line_y_positions, recursion_depth + 1)

    return annot

def main():
    parser = argparse.ArgumentParser(description="Generate key‑metric line charts")
    parser.add_argument("--input", required=True, help="CSV or Excel with Year column")
    parser.add_argument("--font", default=None, help="OTF/TTF font for CJK (e.g. SourceHanSansSC-Bold.otf)")
    parser.add_argument("--output", default=None, help="Output directory for PNGs (defaults to <input_filename>)")
    parser.add_argument("--zip", default="metric_charts.zip", help="Name of ZIP archive")
    parser.add_argument("--theme", choices=['dark', 'light'], default='dark', help="Chart theme: dark or light")
    parser.add_argument("--language", choices=['english', 'chinese'], default='english', help="Output language: english or chinese")
    args = parser.parse_args()

    # Set output directory to <input_filename> (without extension) if not specified
    if args.output is None:
        args.output = os.path.splitext(os.path.basename(args.input))[0]

    if args.font and os.path.exists(args.font):
        fm.fontManager.addfont(args.font)
        plt.rcParams["font.family"] = fm.FontProperties(fname=args.font).get_name()

    df = load_table(args.input)
    if "Year" not in df.columns:
        sys.exit("Input file must have a 'Year' column as first column.")

    years = df["Year"].tolist()

    # Apply theme settings
    valid_rcparams = {k: v for k, v in THEMES[args.theme].items() 
                     if k in ['figure.facecolor', 'axes.facecolor', 'axes.edgecolor', 
                             'axes.labelcolor', 'xtick.color', 'ytick.color']}
    plt.rcParams.update({
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        **valid_rcparams
    })

    # Ensure output dir
    os.makedirs(args.output, exist_ok=True)

    idx = 1
    for col in df.columns:
        if col == "Year":
            continue

        # Split English | Chinese if provided
        if "|" in col:
            eng, chi = col.split("|", 1)
        else:
            eng, chi = col, col  # Use English for both if no Chinese part
        data = df[col]

        # Select title and filename based on language
        title_text = chi.strip() if args.language == 'chinese' and chi != col else eng.strip()
        fname = sanitize_filename(f"{idx}. {title_text}") + ".png"

        # Convert years to integers and clean data
        cleaned_years, cleaned_data = [], []
        for x, y in zip(years, data):
            try:
                x_int = int(x)  # Convert year to integer
                if isinstance(y, str):
                    y = y.replace('%', '').replace('$', '').replace(',', '').strip()
                y_float = float(y)
                if not (y_float != y_float):  # filter NaN
                    cleaned_years.append(x_int)
                    cleaned_data.append(y_float)
            except (ValueError, TypeError):
                continue

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(cleaned_years, cleaned_data, marker='o', color=THEMES[args.theme]['line_color'], 
                linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

        annotations = []
        line_y_positions = [y for y in cleaned_data]  # Y-positions of the plotted line
        if cleaned_data:
            # Average line
            avg = sum(cleaned_data) / len(cleaned_data)
            ax.axhline(avg, color=THEMES[args.theme]['avg_line_color'], linestyle='--', linewidth=2)
            line_y_positions.append(avg)
            annotations.append(ax.text(0.99, avg, f"{LABELS[args.language]['avg']}: {avg:.2f}", 
                                      color=THEMES[args.theme]['avg_line_color'], fontsize=25,
                                      ha='right', va='bottom', transform=ax.get_yaxis_transform()))

            # Buffett's standard line
            buffett_metric, buffett_threshold, buffett_direction = get_buffett_standard(eng)
            if buffett_threshold is not None:
                ax.axhline(buffett_threshold, color=THEMES[args.theme]['buffett_line_color'], 
                          linestyle=':', linewidth=2)
                line_y_positions.append(buffett_threshold)
                annotations.append(ax.text(0.99, buffett_threshold, 
                                          f"{LABELS[args.language]['buffett']} ({buffett_direction}{buffett_threshold:.2f})", 
                                          color=THEMES[args.theme]['buffett_line_color'], fontsize=25,
                                          ha='right', va='top', transform=ax.get_yaxis_transform()))

            # Annotate latest value
            latest_x, latest_y = cleaned_years[-1], cleaned_data[-1]
            latest_annot = adjust_annotation_position(ax, fig, latest_x, latest_y, 
                                                     f"{LABELS[args.language]['latest']}: {latest_y:.2f}", 
                                                     THEMES[args.theme]['latest_annot_color'], 
                                                     offset_x=10, offset_y=0, avoid=annotations, 
                                                     line_y_positions=line_y_positions)
            annotations.append(latest_annot)

            # Annotate max value
            max_idx = cleaned_data.index(max(cleaned_data))
            max_x, max_y = cleaned_years[max_idx], cleaned_data[max_idx]
            offset_y = ANNOTATION_OFFSET if abs(max_y - latest_y) < MIN_DISTANCE else 0
            max_annot = adjust_annotation_position(ax, fig, max_x, max_y, 
                                                  f"{LABELS[args.language]['max']}: {max_y:.2f}", 
                                                  THEMES[args.theme]['max_annot_color'], 
                                                  offset_x=10, offset_y=offset_y, avoid=annotations, 
                                                  line_y_positions=line_y_positions)
            annotations.append(max_annot)

            # Annotate min value
            min_idx = cleaned_data.index(min(cleaned_data))
            min_x, min_y = cleaned_years[min_idx], cleaned_data[min_idx]
            offset_y = ANNOTATION_OFFSET if abs(min_y - latest_y) < MIN_DISTANCE or abs(min_y - max_y) < MIN_DISTANCE else 0
            min_annot = adjust_annotation_position(ax, fig, min_x, min_y, 
                                                  f"{LABELS[args.language]['min']}: {min_y:.2f}", 
                                                  THEMES[args.theme]['min_annot_color'], 
                                                  offset_x=10, offset_y=offset_y, avoid=annotations, 
                                                  line_y_positions=line_y_positions)
            annotations.append(min_annot)

        ax.set_title(title_text, color=THEMES[args.theme]['title_color'], pad=20)

        ax.set_xlabel(LABELS[args.language]['xlabel'], color=THEMES[args.theme]['axes.labelcolor'])
        ax.set_ylabel(detect_ylabel(eng, args.language), color=THEMES[args.theme]['axes.labelcolor'])

        # Apply percentage formatter for rate metrics
        if any(rate in eng for rate in [
            'Revenue Growth Rate', 'Gross Margin', 'Operation Margin',
            'Net Income Growth Rate', 'Net Income Margin', 'Free Cash Flow Margin',
            'Owner\'s-Earnings Margin', 'Return on Retained Earnings',
            'Return On Invested Capital (ROIC)', 'Return on Equity (ROE)',
            'Return on Assets (ROA)', 'Owner\'s-Earnings/Total Market Cap',
            'Interest Expense / Operating Income', 'Depreciation / Gross Profit'
        ]) or '%' in eng:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        # Axis styling
        ax.tick_params(colors=THEMES[args.theme]['xtick.color'])
        for spine in ax.spines.values():
            spine.set_color(THEMES[args.theme]['spine_color'])
        ax.margins(x=0.05)

        fig.savefig(os.path.join(args.output, fname), dpi=100, bbox_inches='tight')
        plt.close(fig)
        gc.collect()
        idx += 1

    # Zip results
    with ZipFile(args.zip, 'w') as z:
        for f in sorted(os.listdir(args.output)):
            z.write(os.path.join(args.output, f), arcname=f)
    print(f"✅ Charts saved to '{args.output}/' and zipped into '{args.zip}'")

if __name__ == "__main__":
    main()