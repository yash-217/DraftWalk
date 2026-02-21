import { useAppStore } from '../store/useAppStore';
import type { CameraView } from '../types';
import { useRef } from 'react';

const VIEWS: { key: CameraView; icon: string; label: string }[] = [
    { key: 'perspective', icon: '🎥', label: 'Perspective' },
    { key: 'isometric', icon: '◇', label: 'Isometric' },
    { key: 'top', icon: '⬆', label: 'Top' },
    { key: 'front', icon: '▶', label: 'Front' },
    { key: 'back', icon: '◀', label: 'Back' },
    { key: 'left', icon: '⏴', label: 'Left' },
    { key: 'right', icon: '⏵', label: 'Right' },
    { key: 'walkthrough', icon: '🚶', label: 'Walk Mode' },
];

export function Sidebar() {
    const { isSidebarOpen, cameraView, setCameraView, setScene, setLoading } = useAppStore();
    const objects = useAppStore((s) => s.scene.objects);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        if (file.type === 'application/json') {
            try {
                const text = await file.text();
                setScene(JSON.parse(text));
            } catch {
                alert('Invalid JSON draft file');
            }
        } else {
            setLoading(true);
            try {
                const fd = new FormData();
                fd.append('file', file);
                const res = await fetch('/api/process', { method: 'POST', body: fd });
                if (res.ok) setScene(await res.json());
            } catch {
                alert('Failed to process image. Make sure the Py backend is running.');
            } finally {
                setLoading(false);
            }
        }
        e.target.value = '';
    };

    return (
        <aside className={`sidebar ${!isSidebarOpen ? 'sidebar--collapsed' : ''}`}>
            <div className="sidebar__section">
                <div className="sidebar__heading">Import</div>
                <div className="upload-zone" onClick={() => fileInputRef.current?.click()}>
                    <div className="upload-zone__icon">📐</div>
                    <div style={{ fontSize: '0.8rem', color: '#7a7470' }}>Upload floor plan or draft</div>
                </div>
                <input ref={fileInputRef} type="file" accept="image/*,.json" onChange={handleFileChange} style={{ display: 'none' }} />
            </div>

            <div className="sidebar__divider" />

            <div className="sidebar__section">
                <div className="sidebar__heading">Views</div>
                <ul className="sidebar__item-list">
                    {VIEWS.map((v) => (
                        <li key={v.key}>
                            <button
                                className={`sidebar__item ${cameraView === v.key ? 'sidebar__item--active' : ''}`}
                                onClick={() => setCameraView(v.key)}
                            >
                                <span className="sidebar__item-icon" style={{ width: 20, textAlign: 'center' }}>{v.icon}</span>
                                {v.label}
                            </button>
                        </li>
                    ))}
                </ul>
            </div>

            <div className="sidebar__divider" />

            <div className="sidebar__section">
                <div className="sidebar__heading">Objects ({objects.length})</div>
                <ul className="sidebar__item-list">
                    {objects.map((obj) => (
                        <li key={obj.id} className="sidebar__item" style={{ padding: '4px 8px' }}>
                            <span style={{ width: 10, height: 10, borderRadius: 2, background: obj.color, flexShrink: 0 }} />
                            <span style={{ fontSize: '0.8rem' }}>{obj.label}</span>
                            <span style={{ marginLeft: 'auto', fontSize: '0.7rem', color: '#a09890' }}>{obj.type}</span>
                        </li>
                    ))}
                </ul>
            </div>
        </aside>
    );
}
