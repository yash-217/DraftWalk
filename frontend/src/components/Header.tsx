import { useAppStore } from '../store/useAppStore';

export function Header() {
    const sceneName = useAppStore((s) => s.scene.metadata.name);
    const toggleSidebar = useAppStore((s) => s.toggleSidebar);
    const toggleSettings = useAppStore((s) => s.toggleSettings);
    const toggleChat = useAppStore((s) => s.toggleChat);
    const clearScene = useAppStore((s) => s.clearScene);
    const clearPromptHistory = useAppStore((s) => s.clearPromptHistory);

    const handleExportImage = () => {
        const canvas = document.querySelector('canvas');
        if (!canvas) return;
        const link = document.createElement('a');
        link.download = `${sceneName.replace(/\\s+/g, '_')}_export.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    };

    const handleExportScene = () => {
        const scene = useAppStore.getState().scene;
        const blob = new Blob([JSON.stringify(scene, null, 2)], { type: 'application/json' });
        const link = document.createElement('a');
        link.download = `${sceneName.replace(/\\s+/g, '_')}_draft.json`;
        link.href = URL.createObjectURL(blob);
        link.click();
        URL.revokeObjectURL(link.href);
    };

    const handleClearScene = () => {
        clearScene();
        clearPromptHistory();
    };

    return (
        <header className="header">
            <div className="header__brand">
                <button className="btn btn--icon" onClick={toggleSidebar} title="Toggle sidebar">☰</button>
                <div className="header__logo">D</div>
                <span className="header__title">DraftWalk</span>
                <span className="header__scene-name">{sceneName}</span>
            </div>
            <div className="header__actions">
                <button className="btn btn--sm btn--danger" onClick={handleClearScene} title="Clear the current scene">🗑 Clear</button>
                <button className="btn btn--sm" onClick={handleExportImage}>📷 Screenshot</button>
                <button className="btn btn--sm" onClick={handleExportScene}>💾 Save Draft</button>
                <button className="btn btn--sm" onClick={toggleSettings}>⚙️ Settings</button>
                <button className="btn btn--icon" onClick={toggleChat} title="Toggle AI chat">💬</button>
            </div>
        </header>
    );
}
