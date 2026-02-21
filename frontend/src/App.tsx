import './styles/index.css';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { Viewport } from './components/Viewport';
import { SettingsPanel } from './components/SettingsPanel';
import { PromptPanel } from './components/PromptPanel';

export default function App() {
    return (
        <div className="app-layout">
            <Header />
            <div className="app-body">
                <Sidebar />
                <Viewport />
                <PromptPanel />
            </div>
            <SettingsPanel />
        </div>
    );
}
