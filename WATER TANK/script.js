'use strict';

function trapWaterTotal(heights) {
  if (!Array.isArray(heights) || heights.length < 3) return 0;

  let left = 0;
  let right = heights.length - 1;
  let leftMax = 0;
  let rightMax = 0;
  let total = 0;

  while (left < right) {
    if (heights[left] <= heights[right]) {
      leftMax = Math.max(leftMax, heights[left]);
      total += leftMax - heights[left];
      left++;
    } else {
      rightMax = Math.max(rightMax, heights[right]);
      total += rightMax - heights[right];
      right--;
    }
  }

  return total;
}

function computeWaterProfile(heights) {
  const n = heights.length;
  if (n === 0) {
    return { levels: [], total: 0, maxHeight: 0 };
  }

  const maxLeft = new Array(n);
  const maxRight = new Array(n);

  maxLeft[0] = heights[0];
  for (let i = 1; i < n; i++) {
    maxLeft[i] = Math.max(maxLeft[i - 1], heights[i]);
  }

  maxRight[n - 1] = heights[n - 1];
  for (let i = n - 2; i >= 0; i--) {
    maxRight[i] = Math.max(maxRight[i + 1], heights[i]);
  }

  const levels = new Array(n);
  let total = 0;
  let maxHeight = 0;

  for (let i = 0; i < n; i++) {
    const waterLevel = Math.min(maxLeft[i], maxRight[i]);
    const water = Math.max(0, waterLevel - heights[i]);
    levels[i] = water;
    total += water;
    maxHeight = Math.max(maxHeight, heights[i], waterLevel);
  }

  return { levels, total, maxHeight };
}

function parseHeights(raw) {
  const trimmed = (raw || '').trim();
  if (!trimmed) {
    return { ok: false, error: 'Enter at least one value, e.g. 0,4,0,0,0,6,0,6,4,0' };
  }

  const tokens = trimmed
    .replace(/[[\]]/g, '')
    .split(/[\s,]+/)
    .filter(Boolean);

  if (tokens.length === 0) {
    return { ok: false, error: 'Enter at least one value, e.g. 0,4,0,0,0,6,0,6,4,0' };
  }

  const values = [];
  for (const token of tokens) {
    if (!/^-?\d+$/.test(token)) {
      return { ok: false, error: `"${token}" is not a whole number.` };
    }
    const num = Number(token);
    if (!Number.isInteger(num) || num < 0) {
      return { ok: false, error: `"${token}" must be greater than -1 (0 or higher).` };
    }
    values.push(num);
  }

  return { ok: true, values };
}

const SVG_NS = 'http://www.w3.org/2000/svg';

function el(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const key in attrs) {
    node.setAttribute(key, attrs[key]);
  }
  return node;
}

function renderSVG(container, heights, levels, maxHeight) {
  container.innerHTML = '';

  const n = heights.length;
  const rows = Math.max(maxHeight, 1);
  const cell = 44;
  const padding = 2;
  const width = n * cell + padding * 2;
  const height = rows * cell + padding * 2;

  const svg = el('svg', {
    viewBox: `0 0 ${width} ${height}`,
    width: '100%',
    role: 'img',
    'aria-label': 'Water tank diagram',
    preserveAspectRatio: 'xMidYMid meet',
  });
  svg.classList.add('tank-svg');

  const defs = el('defs', {});
  const grad = el('linearGradient', { id: 'waterGrad', x1: '0', y1: '0', x2: '0', y2: '1' });
  grad.appendChild(el('stop', { offset: '0%', 'stop-color': '#5FE3F0' }));
  grad.appendChild(el('stop', { offset: '100%', 'stop-color': '#0FA3B1' }));
  defs.appendChild(grad);

  const blockGrad = el('linearGradient', { id: 'blockGrad', x1: '0', y1: '0', x2: '0', y2: '1' });
  blockGrad.appendChild(el('stop', { offset: '0%', 'stop-color': '#FFD766' }));
  blockGrad.appendChild(el('stop', { offset: '100%', 'stop-color': '#F2A93C' }));
  defs.appendChild(blockGrad);
  svg.appendChild(defs);

  // grid
  for (let c = 0; c < n; c++) {
    for (let r = 0; r < rows; r++) {
      const x = padding + c * cell;
      const y = padding + r * cell;
      svg.appendChild(
        el('rect', {
          x, y, width: cell, height: cell,
          class: 'grid-cell',
        })
      );
    }
  }

  // blocks/water
  for (let c = 0; c < n; c++) {
    const blockH = heights[c];
    const waterH = levels[c];
    const x = padding + c * cell;

    if (blockH > 0) {
      const y = padding + (rows - blockH) * cell;
      svg.appendChild(
        el('rect', {
          x, y, width: cell, height: blockH * cell,
          fill: 'url(#blockGrad)',
          class: 'block-cell',
        })
      );
    }

    if (waterH > 0) {
      const y = padding + (rows - blockH - waterH) * cell;
      svg.appendChild(
        el('rect', {
          x, y, width: cell, height: waterH * cell,
          fill: 'url(#waterGrad)',
          class: 'water-cell',
        })
      );
    }

    // label
    const label = el('text', {
      x: x + cell / 2,
      y: height - padding + 16,
      class: 'col-label',
      'text-anchor': 'middle',
    });
    label.textContent = String(heights[c]);
    svg.appendChild(label);
  }

  container.appendChild(svg);
}

function renderTable(container, heights, levels, maxHeight) {
  container.innerHTML = '';

  const n = heights.length;
  const rows = Math.max(maxHeight, 1);

  const table = document.createElement('table');
  table.className = 'tank-table';
  const tbody = document.createElement('tbody');

  for (let r = rows - 1; r >= 0; r--) {
    const tr = document.createElement('tr');
    for (let c = 0; c < n; c++) {
      const td = document.createElement('td');
      const rowFromBottom = rows - r;
      if (rowFromBottom <= heights[c]) {
        td.className = 'cell-block';
      } else if (rowFromBottom <= heights[c] + levels[c]) {
        td.className = 'cell-water';
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }

  table.appendChild(tbody);

  const footRow = document.createElement('tr');
  for (let c = 0; c < n; c++) {
    const td = document.createElement('td');
    td.className = 'cell-label';
    td.textContent = heights[c];
    footRow.appendChild(td);
  }
  const tfoot = document.createElement('tfoot');
  tfoot.appendChild(footRow);
  table.appendChild(tfoot);

  container.appendChild(table);
}

const EXAMPLES = {
  'reference': [0, 4, 0, 0, 0, 6, 0, 6, 4, 0],
  'classic': [4, 2, 0, 3, 2, 5],
  'flat': [3, 3, 3, 3],
  'no-water': [1, 2, 3, 4, 5],
  'single-well': [5, 0, 5],
};

function randomHeights() {
  const n = 6 + Math.floor(Math.random() * 9);
  const arr = [];
  for (let i = 0; i < n; i++) {
    arr.push(Math.floor(Math.random() * 8));
  }
  return arr;
}

function init() {
  const input = document.getElementById('heights-input');
  const form = document.getElementById('tank-form');
  const errorBox = document.getElementById('error-box');
  const totalReadout = document.getElementById('total-readout');
  const diagram = document.getElementById('diagram');
  const exampleSelect = document.getElementById('example-select');
  const randomBtn = document.getElementById('random-btn');
  const viewToggle = document.getElementById('view-toggle');

  let currentView = 'svg';

  function setError(msg) {
    errorBox.textContent = msg || '';
    errorBox.hidden = !msg;
  }

  function render() {
    const parsed = parseHeights(input.value);
    if (!parsed.ok) {
      setError(parsed.error);
      totalReadout.textContent = '—';
      diagram.innerHTML = '';
      return;
    }
    setError('');

    const { values } = parsed;
    const { levels, total, maxHeight } = computeWaterProfile(values);

    // cross-check
    const crossCheck = trapWaterTotal(values.slice());
    if (crossCheck !== total) {
      console.warn('Water total mismatch between algorithms', { crossCheck, total });
    }

    totalReadout.textContent = `${total} unit${total === 1 ? '' : 's'}`;

    if (values.length === 0) {
      diagram.innerHTML = '';
      return;
    }

    if (currentView === 'svg') {
      renderSVG(diagram, values, levels, Math.max(maxHeight, 1));
    } else {
      renderTable(diagram, values, levels, Math.max(maxHeight, 1));
    }
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    render();
  });

  input.addEventListener('input', () => {
    render();
  });

  exampleSelect.addEventListener('change', () => {
    const key = exampleSelect.value;
    if (key && EXAMPLES[key]) {
      input.value = EXAMPLES[key].join(',');
      render();
    }
  });

  randomBtn.addEventListener('click', () => {
    exampleSelect.value = '';
    input.value = randomHeights().join(',');
    render();
  });

  viewToggle.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-view]');
    if (!btn) return;
    currentView = btn.dataset.view;
    for (const b of viewToggle.querySelectorAll('button[data-view]')) {
      b.classList.toggle('active', b === btn);
    }
    render();
  });

  input.value = EXAMPLES.reference.join(',');
  render();
}

document.addEventListener('DOMContentLoaded', init);

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { trapWaterTotal, computeWaterProfile, parseHeights };
}