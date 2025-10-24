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

  return (
    <div className="bg-white p-4 rounded shadow mt-4">
      <h3 className="text-lg font-semibold mb-2">Association Rules</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mb-3">
        <div>
          <label className="block text-sm">Min Support</label>
          <input type="number" step="0.01" min="0" max="1" value={minSupport}
                 onChange={e => setMinSupport(e.target.value)} className="w-full p-2 border rounded" />
        </div>
        <div>
          <label className="block text-sm">Min Confidence</label>
          <input type="number" step="0.01" min="0" max="1" value={minThreshold}
                 onChange={e => setMinThreshold(e.target.value)} className="w-full p-2 border rounded" />
        </div>
        <div className="flex items-end">
          <button onClick={runApriori} className="bg-indigo-600 text-white px-3 py-2 rounded">Run Apriori</button>
        </div>
      </div>

      {evaluation && (
        <div className="mt-3">
          <h4 className="font-medium mb-2">Evaluation Summary</h4>
          <div className="text-sm">
            <div><strong>Number of rules:</strong> {evaluation.n_rules}</div>
            <div><strong>Number of frequent itemsets:</strong> {evaluation.n_frequent_itemsets}</div>
            <div><strong>Average support:</strong> {evaluation.avg_support ? (evaluation.avg_support*100).toFixed(2)+'%' : 'n/a'}</div>
            <div><strong>Average confidence:</strong> {evaluation.avg_confidence ? (evaluation.avg_confidence*100).toFixed(2)+'%' : 'n/a'}</div>
            <div><strong>Average lift:</strong> {evaluation.avg_lift ? evaluation.avg_lift.toFixed(3) : 'n/a'}</div>
          </div>
          {modelPath && (
            <div className="mt-2">
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
                className="bg-gray-700 text-white px-3 py-1 rounded"
              >
                Download Rules
              </button>
            </div>
          )}
        </div>
      )}

      {loading && <div className="text-sm text-blue-600">Running association mining...</div>}
      {error && <div className="text-sm text-red-600">{error}</div>}

      {itemsets && itemsets.length > 0 && (
        <div className="mt-3">
          <h4 className="font-medium mb-2">Frequent Itemsets</h4>
          <div className="overflow-x-auto text-sm">
            <table className="min-w-full">
              <thead>
                <tr>
                  <th className="px-2 py-1 bg-gray-100">support</th>
                  <th className="px-2 py-1 bg-gray-100">itemsets</th>
                </tr>
              </thead>
              <tbody>
                {itemsets.map((it, idx) => (
                  <tr key={idx} className="border-t">
                    <td className="px-2 py-1">{it.support?.toFixed ? (it.support * 100).toFixed(2) + '%' : String(it.support)}</td>
                    <td className="px-2 py-1">{JSON.stringify((it.itemsets || it['itemsets'] || it['items'] || it).map(x => String(x).replace(/^\d+=/, '')))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {rules && rules.length > 0 && (
        <div className="mt-3">
          <h4 className="font-medium mb-2">Association Rules</h4>
          <div className="overflow-x-auto text-sm">
            <table className="min-w-full">
              <thead>
                <tr>
                  <th className="px-2 py-1 bg-gray-100">antecedents</th>
                  <th className="px-2 py-1 bg-gray-100">consequents</th>
                  <th className="px-2 py-1 bg-gray-100">support</th>
                  <th className="px-2 py-1 bg-gray-100">confidence</th>
                  <th className="px-2 py-1 bg-gray-100">lift</th>
                </tr>
              </thead>
              <tbody>
                {rules.map((r, idx) => (
                  <tr key={idx} className="border-t">
                    <td className="px-2 py-1">{JSON.stringify((r.antecedents || r['antecedents'] || r['antecedent'] || []).map(x => String(x).replace(/^\d+=/, '')))}</td>
                    <td className="px-2 py-1">{JSON.stringify((r.consequents || r['consequents'] || r['consequent'] || []).map(x => String(x).replace(/^\d+=/, '')))}</td>
                    <td className="px-2 py-1">{r.support?.toFixed ? (r.support * 100).toFixed(2) + '%' : String(r.support)}</td>
                    <td className="px-2 py-1">{r.confidence?.toFixed ? (r.confidence * 100).toFixed(2) + '%' : String(r.confidence)}</td>
                    <td className="px-2 py-1">{r.lift?.toFixed ? r.lift.toFixed(4) : String(r.lift)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Show top items summary even if no itemsets/rules */}
      {topItems && topItems.length > 0 && (
        <div className="mt-4">
          <h4 className="font-medium mb-2">Top Items (by count)</h4>
          <div className="overflow-x-auto text-sm">
            <table className="min-w-full">
              <thead>
                <tr>
                  <th className="px-2 py-1 bg-gray-100">item</th>
                  <th className="px-2 py-1 bg-gray-100">count</th>
                  <th className="px-2 py-1 bg-gray-100">support</th>
                </tr>
              </thead>
              <tbody>
                {topItems.map((it, idx) => (
                  <tr key={idx} className="border-t">
                    <td className="px-2 py-1">{it.item}</td>
                    <td className="px-2 py-1">{it.count}</td>
                    <td className="px-2 py-1">{(it.support * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {topPairs && topPairs.length > 0 && (
        <div className="mt-4">
          <h4 className="font-medium mb-2">Top Pairs</h4>
          <div className="overflow-x-auto text-sm">
            <table className="min-w-full">
              <thead>
                <tr>
                  <th className="px-2 py-1 bg-gray-100">item A</th>
                  <th className="px-2 py-1 bg-gray-100">item B</th>
                  <th className="px-2 py-1 bg-gray-100">count</th>
                  <th className="px-2 py-1 bg-gray-100">support</th>
                  <th className="px-2 py-1 bg-gray-100">A→B conf</th>
                  <th className="px-2 py-1 bg-gray-100">B→A conf</th>
                  <th className="px-2 py-1 bg-gray-100">A→B lift</th>
                </tr>
              </thead>
              <tbody>
                {topPairs.map((p, idx) => (
                  <tr key={idx} className="border-t">
                    <td className="px-2 py-1">{p.pair[0]}</td>
                    <td className="px-2 py-1">{p.pair[1]}</td>
                    <td className="px-2 py-1">{p.count}</td>
                    <td className="px-2 py-1">{(p.support * 100).toFixed(2)}%</td>
                    <td className="px-2 py-1">{(p.conf_a_b * 100).toFixed(2)}%</td>
                    <td className="px-2 py-1">{(p.conf_b_a * 100).toFixed(2)}%</td>
                    <td className="px-2 py-1">{p.lift_a_b.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Network visualization */}
      <div className="mt-6">
        <h4 className="font-medium mb-2">Association Graph</h4>
        <AssociationNetwork {...buildGraph()} />
      </div>

      {/* Summary if no itemsets/rules */}
      {summary && summary.n_itemsets === 0 && summary.n_rules === 0 && (
        <div className="mt-3 text-sm text-gray-600">No multi-item frequent itemsets or rules found with the selected thresholds. Showing top single items instead.</div>
      )}
    </div>
  );
};

export default AssociationRules;
