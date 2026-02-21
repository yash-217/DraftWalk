import { useAppStore } from '../store/useAppStore';

const RESOLUTION_OPTIONS: { label: string; value: 0.5 | 0.75 | 1 | 1.5 | 2 }[] = [
    { label: '0.5×', value: 0.5 },
    { label: '0.75×', value: 0.75 },
    { label: '1×', value: 1 },
    { label: '1.5×', value: 1.5 },
    { label: '2×', value: 2 },
];

export function SettingsPanel() {
    const { isSettingsOpen, toggleSettings, settings, updateSettings } = useAppStore();

    if (!isSettingsOpen) return null;

    return (
        <div className="settings-overlay" onClick={toggleSettings}>
            <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 20 }}>
                    <h2 style={{ fontSize: '1.2rem', fontWeight: 600 }}>Settings</h2>
                    <button onClick={toggleSettings} style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.2rem' }}>✕</button>
                </div>

                <div>
                    <div style={{ fontSize: '0.9rem', fontWeight: 500, marginBottom: 8 }}>Resolution Scale</div>
                    <div className="scale-options">
                        {RESOLUTION_OPTIONS.map((opt) => (
                            <button
                                key={opt.value}
                                className={`scale-options__btn ${settings.resolutionScale === opt.value ? 'scale-options__btn--active' : ''}`}
                                onClick={() => updateSettings({ resolutionScale: opt.value })}
                            >
                                {opt.label}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="toggle">
                    <div style={{ fontSize: '0.9rem', fontWeight: 500 }}>Anti-Aliasing</div>
                    <button className={`toggle__switch ${settings.antiAliasing ? 'toggle__switch--on' : ''}`} onClick={() => updateSettings({ antiAliasing: !settings.antiAliasing })} />
                </div>

                <div className="toggle">
                    <div style={{ fontSize: '0.9rem', fontWeight: 500 }}>Shadows</div>
                    <button className={`toggle__switch ${settings.shadows ? 'toggle__switch--on' : ''}`} onClick={() => updateSettings({ shadows: !settings.shadows })} />
                </div>
            </div>
        </div>
    );
}
