import type { SceneGraph } from '../types';

export const demoScene: SceneGraph = {
    metadata: { name: 'Demo Apartment', units: 'meters', source: 'built-in' },
    walls: [
        { id: 'w1', start: { x: 0, y: 0, z: 0 }, end: { x: 10, y: 0, z: 0 }, height: 2.8, thickness: 0.2, color: '#e8e0d4' },
        { id: 'w2', start: { x: 10, y: 0, z: 0 }, end: { x: 10, y: 0, z: 8 }, height: 2.8, thickness: 0.2, color: '#e8e0d4' },
        { id: 'w3', start: { x: 10, y: 0, z: 8 }, end: { x: 0, y: 0, z: 8 }, height: 2.8, thickness: 0.2, color: '#e8e0d4' },
        { id: 'w4', start: { x: 0, y: 0, z: 8 }, end: { x: 0, y: 0, z: 0 }, height: 2.8, thickness: 0.2, color: '#e8e0d4' },
        { id: 'w5', start: { x: 5, y: 0, z: 0 }, end: { x: 5, y: 0, z: 5 }, height: 2.8, thickness: 0.15, color: '#f0ebe3' },
        { id: 'w6', start: { x: 0, y: 0, z: 5 }, end: { x: 5, y: 0, z: 5 }, height: 2.8, thickness: 0.15, color: '#f0ebe3' },
    ],
    floors: [
        {
            id: 'f1',
            vertices: [{ x: 0, y: 0, z: 0 }, { x: 10, y: 0, z: 0 }, { x: 10, y: 0, z: 8 }, { x: 0, y: 0, z: 8 }],
            color: '#d4c9b8',
            material: 'wood',
        },
    ],
    objects: [
        { id: 'obj1', type: 'furniture', label: 'Sofa', position: { x: 7.5, y: 0.4, z: 2 }, rotation: { x: 0, y: 0, z: 0 }, scale: { x: 2, y: 0.8, z: 0.9 }, color: '#6b8f71', geometry: 'box' },
        { id: 'obj2', type: 'furniture', label: 'Coffee Table', position: { x: 7.5, y: 0.25, z: 3.5 }, rotation: { x: 0, y: 0, z: 0 }, scale: { x: 1.2, y: 0.5, z: 0.6 }, color: '#8b6f47', geometry: 'box' },
        { id: 'obj3', type: 'fixture', label: 'Kitchen Counter', position: { x: 7.5, y: 0.45, z: 7 }, rotation: { x: 0, y: 0, z: 0 }, scale: { x: 3, y: 0.9, z: 0.6 }, color: '#c4b5a0', geometry: 'box' },
        { id: 'obj4', type: 'furniture', label: 'Bed', position: { x: 2.5, y: 0.3, z: 2.5 }, rotation: { x: 0, y: 0, z: 0 }, scale: { x: 1.6, y: 0.6, z: 2 }, color: '#a8c4d4', geometry: 'box' },
        { id: 'obj5', type: 'door', label: 'Bedroom Door', position: { x: 5, y: 1, z: 4 }, rotation: { x: 0, y: Math.PI / 2, z: 0 }, scale: { x: 0.9, y: 2, z: 0.05 }, color: '#8b6f47', geometry: 'box' },
        { id: 'obj6', type: 'window', label: 'Living Room Window', position: { x: 10, y: 1.5, z: 4 }, rotation: { x: 0, y: Math.PI / 2, z: 0 }, scale: { x: 1.5, y: 1.2, z: 0.05 }, color: '#b8d8e8', geometry: 'box' },
    ],
};
