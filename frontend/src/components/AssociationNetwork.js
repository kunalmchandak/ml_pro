import React, { useEffect, useRef } from 'react';

const AssociationNetwork = ({ nodes = [], edges = [], options = {} }) => {
  const containerRef = useRef(null);
  const networkRef = useRef(null);

  useEffect(() => {
    if (!window.vis || !containerRef.current) return;

    const visNodes = new window.vis.DataSet(nodes.map((n, i) => ({ id: n.id ?? i, label: n.label, value: n.value })));
    const visEdges = new window.vis.DataSet(edges.map((e, i) => ({ id: i, from: e.from, to: e.to, value: e.weight })));

    const data = { nodes: visNodes, edges: visEdges };
    const defaultOptions = {
      nodes: { shape: 'dot', scaling: { min: 5, max: 30 }, font: { size: 12 } },
      edges: { color: { inherit: true }, smooth: { enabled: true } },
      physics: { stabilization: true, barnesHut: { gravitationalConstant: -2000 } }
    };

    networkRef.current = new window.vis.Network(containerRef.current, data, { ...defaultOptions, ...options });

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [nodes, edges, options]);

  return <div ref={containerRef} style={{ width: '100%', height: '400px' }} />;
};

export default AssociationNetwork;
