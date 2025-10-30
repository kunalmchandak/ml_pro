import React, { useEffect, useRef } from 'react';

const AssociationNetwork = ({ nodes = [], edges = [], options = {} }) => {
  const containerRef = useRef(null);
  const networkRef = useRef(null);

  useEffect(() => {
    if (!window.vis || !containerRef.current) return;

    const visNodes = new window.vis.DataSet(nodes.map((n, i) => ({ 
      id: n.id ?? i, 
      label: n.label, 
      value: n.value,
      color: {
        background: '#9333EA', // purple-600
        border: '#7C3AED', // purple-700
        highlight: {
          background: '#7C3AED',
          border: '#6D28D9'
        }
      },
      font: {
        color: '#1F2937', // gray-800
        size: 14,
        face: 'Inter'
      }
    })));
    
    const visEdges = new window.vis.DataSet(edges.map((e, i) => ({ 
      id: i, 
      from: e.from, 
      to: e.to, 
      value: e.weight,
      color: {
        color: '#818CF8', // indigo-400
        highlight: '#6366F1', // indigo-500
        hover: '#6366F1' // indigo-500
      }
    })));

    const data = { nodes: visNodes, edges: visEdges };
    const defaultOptions = {
      nodes: { 
        shape: 'dot', 
        scaling: { min: 8, max: 35 },
        shadow: {
          enabled: true,
          color: 'rgba(147, 51, 234, 0.2)', // purple-600 with opacity
          size: 5,
          x: 0,
          y: 2
        }
      },
      edges: { 
        smooth: { 
          enabled: true,
          type: 'continuous'
        },
        width: 2,
        shadow: {
          enabled: true,
          color: 'rgba(99, 102, 241, 0.2)', // indigo-500 with opacity
          size: 3,
          x: 0,
          y: 2
        }
      },
      physics: { 
        stabilization: { 
          enabled: true,
          iterations: 100,
          fit: true 
        }, 
        barnesHut: { 
          gravitationalConstant: -2000,
          springLength: 150,
          springConstant: 0.04
        } 
      },
      interaction: {
        hover: true,
        tooltipDelay: 200
      }
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
