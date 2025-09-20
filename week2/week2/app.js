// Expected columns for Titanic (used only to validate correct parsing).
// If you swap datasets later, change this list accordingly.
const EXPECTED_MIN = ['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'];
// Having 'Name' present is a strong signal that quotes around commas were handled correctly.
const NICE_TO_HAVE = ['Name'];

// Try multiple parse configs and score them; pick the best.
async function robustParseCSV(file, manualDelim, manualQuote){
  const tryDelims = manualDelim && manualDelim !== 'auto' ? [manualDelim] : [',',';','\t'];
  const tryQuotes = manualQuote && manualQuote !== 'auto' ? [manualQuote] : ['"', "'"];

  let best = {score: -Infinity, rows: [], cfg: null, diag: ''};

  for(const d of tryDelims){
    for(const q of tryQuotes){
      const rows = await parseWithPapa(file, d, q);
      const {score, diag} = scoreParsed(rows, d, q);
      if(score > best.score){
        best = {score, rows, cfg: {delimiter: d, quoteChar: q}, diag};
      }
    }
  }
  return best;
}

function parseWithPapa(file, delimiter, quoteChar){
  return new Promise((resolve,reject)=>{
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: 'greedy',
      delimiter,           // ',', ';', or '\t'
      quoteChar,           // '"' or '\''
      // escapeChar left default (same as quoteChar)
      complete: res => resolve(res.data),
      error: reject
    });
  });
}

// Score a parse: presence of expected columns, mode row width consistency, and low missingness.
function scoreParsed(rows, delimiter, quoteChar){
  if(!rows || !rows.length) return {score: -1e9, diag: 'No rows'};
  const headers = Object.keys(rows[0] ?? {});
  const headerSet = new Set(headers);

  // 1) Expected columns present?
  let have = 0; for(const c of EXPECTED_MIN) if(headerSet.has(c)) have++;
  const expectedHit = have/EXPECTED_MIN.length; // 0..1

  // 2) Row width consistency
  const widthCounts = {};
  for(const r of rows){ const w = Object.keys(r).length; widthCounts[w] = (widthCounts[w]||0)+1; }
  const modeW = Object.entries(widthCounts).sort((a,b)=>b[1]-a[1])[0];
  const consistency = modeW ? modeW[1]/rows.length : 0; // 0..1

  // 3) Nice to have (e.g., 'Name' when commas exist)
  const bonus = NICE_TO_HAVE.every(c => headerSet.has(c)) ? 0.1 : 0;

  // 4) Rough missingness penalty (lighter weight)
  const missPct = roughMissingPct(rows)/100; // 0..1
  const missPenalty = -0.2*missPct;

  const score = 3*expectedHit + 2*consistency + bonus + missPenalty;

  const diag = `Parsed with delimiter='${humanDelim(delimiter)}', quote='${quoteChar}' | `
             + `expectedHit=${(100*expectedHit).toFixed(0)}% | `
             + `rowConsistency=${(100*consistency).toFixed(0)}% | `
             + (bonus>0 ? 'Name:OK | ' : '') + `missing≈${(missPct*100).toFixed(1)}%`;
  return {score, diag};
}

function humanDelim(d){ return d === '\t' ? '\\t' : d; }

function roughMissingPct(rows){
  if(!rows.length) return 100;
  const cols = Object.keys(rows[0]);
  let miss = 0, total = rows.length * cols.length;
  for(const r of rows){ for(const c of cols){ const v=r[c]; if(v===''||v==null||v===undefined) miss++; } }
  return +(100*miss/total).toFixed(1);
}
