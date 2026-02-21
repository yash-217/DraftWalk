import { Canvas } from '@react-three/fiber';
import { useAppStore } from '../store/useAppStore';
import { SceneRenderer } from './SceneRenderer';
import { CameraController, WalkthroughController } from './CameraController';

export function Viewport() {
    const settings = useAppStore((s) => s.settings);
    const cameraView = useAppStore((s) => s.cameraView);
    const isLoading = useAppStore((s) => s.isLoading);

    return (
        <div className="canvas-area">
            <div className="view-badge">{cameraView}</div>

            <Canvas
                shadows={settings.shadows}
                dpr={settings.resolutionScale}
                gl={{
                    antialias: settings.antiAliasing,
                    preserveDrawingBuffer: true,
                    toneMapping: 3,
                    toneMappingExposure: 1.1,
                }}
                camera={{ fov: 55, near: 0.1, far: 200, position: [12, 8, 12] }}
                style={{ width: '100%', height: '100%' }}
            >
                <CameraController />
                <WalkthroughController />
                <SceneRenderer />
            </Canvas>

            {isLoading && (
                <div className="loading-overlay">
                    <div className="spinner" style={{ width: 36, height: 36, border: '3px solid var(--color-border)', borderTopColor: 'var(--color-primary)', borderRadius: '50%', animation: 'spin 800ms linear infinite' }} />
                    <span style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-muted)', marginTop: '10px' }}>Processing…</span>
                </div>
            )}

            {cameraView === 'walkthrough' && (
                <div style={{ position: 'absolute', top: 12, right: 12, background: 'var(--color-surface)', border: '1px solid var(--color-border)', padding: '6px 14px', borderRadius: 'var(--radius-md)', fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)', boxShadow: 'var(--shadow-sm)', zIndex: 10 }}>
                    Click to look around · WASD to move · Esc to release
                </div>
            )}
        </div>
    );
}
