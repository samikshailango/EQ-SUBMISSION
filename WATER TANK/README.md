# Water Tank 

**Input:** `[0,4,0,0,0,6,0,6,4,0]` → **Output:** `18 units`.

## Run it

Open `index.html` in a browser

```bash
open index.html          
# to serve it (locally):
python3 -m http.server 8080
```

## Files

| File          | Purpose                                                        |
|---------------|------------------------------------------------------------------|
| `index.html`  | Page structure — input field, presets, result readout, diagram |
| `style.css`   | All styling                |
| `script.js`   | Algorithm, input validation, SVG + table rendering, event wiring |
| `test.js`     | Standalone test program with a few test cases    |

## Algorithm

Two independent implementations are used and cross-checked at render time:

1. **Two-pointer sweep** (`trapWaterTotal`) — O(n) time, O(1) extra space.
   Walks from both ends inward, tracking the running max seen from each
   side, and accumulates water based on whichever side has the shorter wall.

2. **Prefix/suffix max profile** (`computeWaterProfile`) — O(n) time, O(n)
   space. For each column `i`:

   ```
   water[i] = max(0, min(maxLeft[i], maxRight[i]) - height[i])
   ```

   This version additionally returns the *per-column* water level, which
   the two-pointer approach doesn't expose — needed to actually draw the
   diagram.


## Input rules

Per the brief, `n` (each block height) must always be greater than -1, i.e.
a non-negative integer. The input field accepts comma- or space-separated
numbers, with or without surrounding brackets (`0,4,0` or `[0, 4, 0]`).
Negative numbers, decimals, and non-numeric tokens are rejected with an
inline error message rather than silently coerced.

## Views

- **Elevation (SVG, default)** — a grid of unit cells; blocks are rendered
  solid, water is rendered above the block up to its resting level. This is
  the diagram style requested in the brief.
- **Table** — the same data rendered as an HTML `<table>`, offered as the
  brief's alternative "Table View."

Toggle between them with the buttons above the diagram.

## Testing

```bash
node test.js
```

This code executes few test cases and displays the result of it.
