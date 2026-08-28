'use strict';

global.document = { addEventListener: () => {} }; 
const { trapWaterTotal, computeWaterProfile, parseHeights } = require('./script.js');

const cases = [
  { input: [0, 4, 0, 0, 0, 6, 0, 6, 4, 0], expected: 18 }, 
  { input: [4, 2, 0, 3, 2, 5], expected: 9 },              
  { input: [3, 3, 3, 3], expected: 0 },                    
  { input: [1, 2, 3, 4, 5], expected: 0 },                  
  { input: [5, 4, 3, 2, 1], expected: 0 },                  
  { input: [5, 0, 5], expected: 5 },                        
  { input: [0, 0, 0], expected: 0 },                        
  { input: [], expected: 0 },                              
  { input: [7], expected: 0 },                              
  { input: [3, 0, 3], expected: 3 },                        
  { input: [2, 0, 2, 0, 2], expected: 4 },                 
];

let failures = 0;

for (const { input, expected } of cases) {
  const total1 = trapWaterTotal(input.slice());
  const { total: total2 } = computeWaterProfile(input.slice());
  const pass = total1 === expected && total2 === expected;
  if (!pass) failures++;
  console.log(
    `${pass ? 'PASS' : 'FAIL'}  [${input.join(',')}]  expected=${expected}  twoPointer=${total1}  profile=${total2}`
  );
}

// Input validation checks
const validationCases = [
  { raw: '0,4,0,0,0,6,0,6,4,0', ok: true },
  { raw: '[0, 4, 0]', ok: true },
  { raw: '-1,2,3', ok: false }, 
  { raw: 'a,b,c', ok: false },
  { raw: '', ok: false },
  { raw: '1.5,2', ok: false },
];

for (const { raw, ok } of validationCases) {
  const result = parseHeights(raw);
  const pass = result.ok === ok;
  if (!pass) failures++;
  console.log(`${pass ? 'PASS' : 'FAIL'}  parseHeights(${JSON.stringify(raw)}) -> ok=${result.ok}`);
}

console.log('\n' + (failures === 0 ? `All ${cases.length + validationCases.length} tests passed.` : `${failures} test(s) FAILED.`));
process.exit(failures === 0 ? 0 : 1);
