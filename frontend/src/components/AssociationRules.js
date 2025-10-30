import React, { useState } from 'react';
import axios from 'axios';
import AssociationNetwork from './AssociationNetwork';

const AssociationRules = ({ selectedDataset, isUserDataset }) => {
  const [minSupport, setMinSupport] = useState(0.1);
  const [minThreshold, setMinThreshold] = useState(0.6);
  const [loading, setLoading] = useState(false);
  const [rules, setRules] = useState([]);
  const [itemsets, setItemsets] = useState([]);
  const [topItems, setTopItems] = useState([]);
  const [summary, setSummary] = useState({});
  const [topPairs, setTopPairs] = useState([]);
  const [error, setError] = useState('');
  const [modelPath, setModelPath] = useState(null);
  const [evaluation, setEvaluation] = useState(null);

  const runApriori = async () => {
    if (!selectedDataset) {
      setError('Select a dataset first');
      return;
    }
    setLoading(true);
    setError('');
    setRules([]);
    setItemsets([]);
    try {
      const payload = {
        algorithm: 'apriori',
        dataset_name: selectedDataset,
        is_user_dataset: !!isUserDataset,
        params: { min_support: Number(minSupport), min_threshold: Number(minThreshold) }
      };
      const resp = await axios.post('http://localhost:8000/train', payload);
  setItemsets(resp.data.frequent_itemsets || []);
  setRules(resp.data.rules || []);
  setTopItems(resp.data.top_items || []);
  setTopPairs(resp.data.top_pairs || []);
  setSummary({ n_itemsets: resp.data.n_frequent_itemsets || 0, n_rules: resp.data.n_rules || 0, n_transactions: resp.data.n_transactions || 0 });
  setModelPath(resp.data.model_path || null);
  setEvaluation(resp.data.evaluation || null);
    } catch (err) {
      console.error(err);
      setError(err?.response?.data?.detail || 'Failed to run association mining');
    } finally {
      setLoading(false);
    }
  };

  // Build nodes and edges for network visualization
  const buildGraph = () => {
    const nodeMap = new Map();
    const edges = [];

    // Add top items as nodes
    topItems.forEach((it, idx) => {
      const label = String(it.item).replace(/^\d+=/, '');
      nodeMap.set(label, { id: label, label, value: it.count });
    });

    // From frequent itemsets, add edges between items in the same itemset
    itemsets.forEach((it) => {
      const items = (it.itemsets || it.items || it['itemsets'] || []).map(x => String(x).replace(/^\d+=/, ''));
      // connect each pair in the itemset
      for (let i = 0; i < items.length; i++) {
        for (let j = i + 1; j < items.length; j++) {
          edges.push({ from: items[i], to: items[j], weight: it.support || 1 });
          nodeMap.set(items[i], nodeMap.get(items[i]) || { id: items[i], label: items[i], value: 1 });
          nodeMap.set(items[j], nodeMap.get(items[j]) || { id: items[j], label: items[j], value: 1 });
        }
      }
    });

    // From rules, add directed edges (antecedent -> consequent)
    rules.forEach((r) => {
      const ants = (r.antecedents || r['antecedents'] || r['antecedent'] || []).map(x => String(x).replace(/^\d+=/, ''));
      const cons = (r.consequents || r['consequents'] || r['consequent'] || []).map(x => String(x).replace(/^\d+=/, ''));
      ants.forEach(a => cons.forEach(c => {
        edges.push({ from: a, to: c, weight: r.confidence || r.lift || 1 });
        nodeMap.set(a, nodeMap.get(a) || { id: a, label: a, value: 1 });
        nodeMap.set(c, nodeMap.get(c) || { id: c, label: c, value: 1 });
      }));
    });

    // Boost edge weights for topPairs so they are visually emphasized
    topPairs.forEach(tp => {
      const [a, b] = tp.pair;
      // find existing edge and increase weight, otherwise add
      const existing = edges.find(e => (e.from === a && e.to === b) || (e.from === b && e.to === a));
      if (existing) {
        existing.weight = Math.max(existing.weight, tp.support || 1) * 2;
      } else {
        edges.push({ from: a, to: b, weight: (tp.support || 0.01) * 2 });
        nodeMap.set(a, nodeMap.get(a) || { id: a, label: a, value: 1 });
        nodeMap.set(b, nodeMap.get(b) || { id: b, label: b, value: 1 });
      }
    });

    const nodes = Array.from(nodeMap.values());
    return { nodes, edges };
  };

  const animationClasses = {
    card: "transform transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:shadow-purple-200/50",
    button: "transition-all duration-300 transform hover:-translate-y-1 hover:scale-105 shadow-lg hover:shadow-xl",
    input: "transition-all duration-200 focus:ring-2 focus:ring-purple-400",
    table: "bg-white/90 backdrop-blur-lg rounded-xl border-2 border-purple-200 overflow-hidden",
    th: "px-4 py-2 bg-gradient-to-r from-purple-600/10 to-indigo-600/10 text-left text-sm font-semibold text-gray-700",
    td: "px-4 py-2 border-t border-purple-100"
  };

  return (
    <div className="bg-white/90 backdrop-blur-lg p-6 rounded-xl border-4 border-purple-400 hover:border-purple-600 transition-all duration-300 shadow-lg">
      <h3 className="text-2xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
        Association Rules
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className={`${animationClasses.card} bg-white/80 p-4 rounded-xl border-2 border-purple-200`}>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Min Support</label>
          <input 
            type="number" 
            step="0.01" 
            min="0" 
            max="1" 
            value={minSupport}
            onChange={e => setMinSupport(e.target.value)} 
            className={`w-full p-3 border-2 border-purple-200 rounded-lg bg-white/80 ${animationClasses.input}`} 
          />
        </div>
        <div className={`${animationClasses.card} bg-white/80 p-4 rounded-xl border-2 border-purple-200`}>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Min Confidence</label>
          <input 
            type="number" 
            step="0.01" 
            min="0" 
            max="1" 
            value={minThreshold}
            onChange={e => setMinThreshold(e.target.value)} 
            className={`w-full p-3 border-2 border-purple-200 rounded-lg bg-white/80 ${animationClasses.input}`} 
          />
        </div>
        <div className="flex items-end">
          <button 
            onClick={runApriori} 
            className={`${animationClasses.button} w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 
              text-white font-bold py-3 px-6 rounded-lg`}
          >
            Run
          </button>
        </div>
      </div>

      {evaluation && (
        <div className={`${animationClasses.card} mt-8 bg-white/80 p-6 rounded-xl border-2 border-purple-200`}>
          <h4 className="text-xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
            Evaluation Summary
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className={`${animationClasses.card} bg-white/90 p-4 rounded-xl border border-purple-100`}>
              <div className="text-sm text-gray-600 mb-1">Rules Found</div>
              <div className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
                {evaluation.n_rules}
              </div>
            </div>
            <div className={`${animationClasses.card} bg-white/90 p-4 rounded-xl border border-purple-100`}>
              <div className="text-sm text-gray-600 mb-1">Frequent Itemsets</div>
              <div className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
                {evaluation.n_frequent_itemsets}
              </div>
            </div>
            <div className={`${animationClasses.card} bg-white/90 p-4 rounded-xl border border-purple-100`}>
              <div className="text-sm text-gray-600 mb-1">Average Support</div>
              <div className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
                {evaluation.avg_support ? (evaluation.avg_support*100).toFixed(2)+'%' : 'n/a'}
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className={`${animationClasses.card} bg-white/90 p-4 rounded-xl border border-purple-100`}>
              <div className="text-sm text-gray-600 mb-1">Average Confidence</div>
              <div className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
                {evaluation.avg_confidence ? (evaluation.avg_confidence*100).toFixed(2)+'%' : 'n/a'}
              </div>
            </div>
            <div className={`${animationClasses.card} bg-white/90 p-4 rounded-xl border border-purple-100`}>
              <div className="text-sm text-gray-600 mb-1">Average Lift</div>
              <div className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
                {evaluation.avg_lift ? evaluation.avg_lift.toFixed(3) : 'n/a'}
              </div>
            </div>
          </div>

          {modelPath && (
            <div className="mt-6">
              <button
                onClick={async () => {
                  try {
                    const url = `http://localhost:8000/download_model/${encodeURIComponent(selectedDataset)}/${encodeURIComponent('apriori')}`;
                    const resp = await axios.get(url, { responseType: 'blob' });
                    const blobUrl = window.URL.createObjectURL(new Blob([resp.data]));
                    const link = document.createElement('a');
                    link.href = blobUrl;
                    link.setAttribute('download', modelPath.split('/').pop());
                    document.body.appendChild(link);
                    link.click();
                    link.parentNode.removeChild(link);
                    window.URL.revokeObjectURL(blobUrl);
                  } catch (e) {
                    console.error('Download failed', e);
                    setError('Download failed');
                  }
                }}
                className={`${animationClasses.button} bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 
                  text-white font-bold py-3 px-6 rounded-lg`}
              >
                Download Rules
              </button>
            </div>
          )}
        </div>
      )}

      {loading && (
        <div className="text-sm font-medium text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600 animate-pulse">
          Running association mining...
        </div>
      )}
      {error && (
        <div className="text-sm font-medium text-red-600 bg-red-50 border border-red-200 rounded-lg p-3">
          {error}
        </div>
      )}

      {itemsets && itemsets.length > 0 && (
        <div className={`${animationClasses.card} mt-8 bg-white/80 p-6 rounded-xl border-2 border-purple-200`}>
          <h4 className="text-xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
            Frequent Itemsets
          </h4>
          <div className="overflow-x-auto">
            <table className={`${animationClasses.table} min-w-full`}>
              <thead>
                <tr>
                  <th className={animationClasses.th}>Support</th>
                  <th className={animationClasses.th}>Itemsets</th>
                </tr>
              </thead>
              <tbody>
                {itemsets.map((it, idx) => (
                  <tr key={idx} className="hover:bg-purple-50/50 transition-colors">
                    <td className={animationClasses.td}>
                      <span className="font-semibold text-purple-600">
                        {it.support?.toFixed ? (it.support * 100).toFixed(2) + '%' : String(it.support)}
                      </span>
                    </td>
                    <td className={animationClasses.td}>
                      <code className="bg-gray-50 px-2 py-1 rounded text-sm">
                        {JSON.stringify((it.itemsets || it['itemsets'] || it['items'] || it).map(x => String(x).replace(/^\d+=/, '')))}
                      </code>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {rules && rules.length > 0 && (
        <div className={`${animationClasses.card} mt-8 bg-white/80 p-6 rounded-xl border-2 border-purple-200`}>
          <h4 className="text-xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
            Association Rules
          </h4>
          <div className="overflow-x-auto">
            <table className={`${animationClasses.table} min-w-full`}>
              <thead>
                <tr>
                  <th className={animationClasses.th}>Antecedents</th>
                  <th className={animationClasses.th}>Consequents</th>
                  <th className={animationClasses.th}>Support</th>
                  <th className={animationClasses.th}>Confidence</th>
                  <th className={animationClasses.th}>Lift</th>
                </tr>
              </thead>
              <tbody>
                {rules.map((r, idx) => (
                  <tr key={idx} className="hover:bg-purple-50/50 transition-colors">
                    <td className={animationClasses.td}>
                      <code className="bg-gray-50 px-2 py-1 rounded text-sm">
                        {JSON.stringify((r.antecedents || r['antecedents'] || r['antecedent'] || []).map(x => String(x).replace(/^\d+=/, '')))}
                      </code>
                    </td>
                    <td className={animationClasses.td}>
                      <code className="bg-gray-50 px-2 py-1 rounded text-sm">
                        {JSON.stringify((r.consequents || r['consequents'] || r['consequent'] || []).map(x => String(x).replace(/^\d+=/, '')))}
                      </code>
                    </td>
                    <td className={`${animationClasses.td} font-semibold text-purple-600`}>
                      {r.support?.toFixed ? (r.support * 100).toFixed(2) + '%' : String(r.support)}
                    </td>
                    <td className={`${animationClasses.td} font-semibold text-indigo-600`}>
                      {r.confidence?.toFixed ? (r.confidence * 100).toFixed(2) + '%' : String(r.confidence)}
                    </td>
                    <td className={`${animationClasses.td} font-semibold`}>
                      {r.lift?.toFixed ? r.lift.toFixed(4) : String(r.lift)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Show top items summary even if no itemsets/rules */}
      {topItems && topItems.length > 0 && (
        <div className={`${animationClasses.card} mt-8 bg-white/80 p-6 rounded-xl border-2 border-purple-200`}>
          <h4 className="text-xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
            Top Items (by count)
          </h4>
          <div className="overflow-x-auto">
            <table className={`${animationClasses.table} min-w-full`}>
              <thead>
                <tr>
                  <th className={animationClasses.th}>Item</th>
                  <th className={animationClasses.th}>Count</th>
                  <th className={animationClasses.th}>Support</th>
                </tr>
              </thead>
              <tbody>
                {topItems.map((it, idx) => (
                  <tr key={idx} className="hover:bg-purple-50/50 transition-colors">
                    <td className={animationClasses.td}>
                      <code className="bg-gray-50 px-2 py-1 rounded text-sm">
                        {it.item}
                      </code>
                    </td>
                    <td className={`${animationClasses.td} font-semibold text-purple-600`}>
                      {it.count}
                    </td>
                    <td className={`${animationClasses.td} font-semibold text-indigo-600`}>
                      {(it.support * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {topPairs && topPairs.length > 0 && (
        <div className={`${animationClasses.card} mt-8 bg-white/80 p-6 rounded-xl border-2 border-purple-200`}>
          <h4 className="text-xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
            Top Pairs
          </h4>
          <div className="overflow-x-auto">
            <table className={`${animationClasses.table} min-w-full`}>
              <thead>
                <tr>
                  <th className={animationClasses.th}>Item A</th>
                  <th className={animationClasses.th}>Item B</th>
                  <th className={animationClasses.th}>Count</th>
                  <th className={animationClasses.th}>Support</th>
                  <th className={animationClasses.th}>A→B Conf</th>
                  <th className={animationClasses.th}>B→A Conf</th>
                  <th className={animationClasses.th}>A→B Lift</th>
                </tr>
              </thead>
              <tbody>
                {topPairs.map((p, idx) => (
                  <tr key={idx} className="hover:bg-purple-50/50 transition-colors">
                    <td className={animationClasses.td}>
                      <code className="bg-gray-50 px-2 py-1 rounded text-sm">
                        {p.pair[0]}
                      </code>
                    </td>
                    <td className={animationClasses.td}>
                      <code className="bg-gray-50 px-2 py-1 rounded text-sm">
                        {p.pair[1]}
                      </code>
                    </td>
                    <td className={`${animationClasses.td} font-semibold text-purple-600`}>
                      {p.count}
                    </td>
                    <td className={`${animationClasses.td} font-semibold text-purple-600`}>
                      {(p.support * 100).toFixed(2)}%
                    </td>
                    <td className={`${animationClasses.td} font-semibold text-indigo-600`}>
                      {(p.conf_a_b * 100).toFixed(2)}%
                    </td>
                    <td className={`${animationClasses.td} font-semibold text-indigo-600`}>
                      {(p.conf_b_a * 100).toFixed(2)}%
                    </td>
                    <td className={`${animationClasses.td} font-semibold`}>
                      {p.lift_a_b.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Network visualization */}
      <div className={`${animationClasses.card} mt-8 bg-white/80 p-6 rounded-xl border-2 border-purple-200`}>
        <h4 className="text-xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
          Association Graph
        </h4>
        <div className={`${animationClasses.card} bg-white/90 rounded-lg overflow-hidden border border-purple-100`}>
          <AssociationNetwork {...buildGraph()} />
        </div>
      </div>

      {/* Summary if no itemsets/rules */}
      {summary && summary.n_itemsets === 0 && summary.n_rules === 0 && (
        <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg text-sm text-yellow-700">
          No multi-item frequent itemsets or rules found with the selected thresholds. Showing top single items instead.
        </div>
      )}
    </div>
  );
};

export default AssociationRules;
