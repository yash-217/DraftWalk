import { create } from 'zustand';
import type { SceneGraph, CameraView, AppSettings, PromptMessage } from '../types';
import { demoScene } from './demoScene';

const emptyScene: SceneGraph = {
    metadata: { name: 'Empty Scene', units: 'meters' },
    walls: [],
    floors: [],
    objects: [],
};

interface AppState {
    scene: SceneGraph;
    setScene: (scene: SceneGraph) => void;
    patchScene: (patch: Partial<SceneGraph>) => void;
    clearScene: () => void;

    cameraView: CameraView;
    setCameraView: (view: CameraView) => void;

    settings: AppSettings;
    updateSettings: (patch: Partial<AppSettings>) => void;

    promptHistory: PromptMessage[];
    addPromptMessage: (msg: PromptMessage) => void;
    clearPromptHistory: () => void;

    isSidebarOpen: boolean;
    toggleSidebar: () => void;
    isChatOpen: boolean;
    toggleChat: () => void;
    isSettingsOpen: boolean;
    toggleSettings: () => void;
    isLoading: boolean;
    setLoading: (v: boolean) => void;
}

export const useAppStore = create<AppState>((set) => ({
    scene: demoScene,
    setScene: (scene) => set({ scene }),
    patchScene: (patch) =>
        set((s) => ({ scene: { ...s.scene, ...patch } })),
    clearScene: () => set({ scene: emptyScene }),

    cameraView: 'perspective',
    setCameraView: (cameraView) => set({ cameraView }),

    settings: {
        resolutionScale: 1,
        antiAliasing: true,
        shadows: true,
    },
    updateSettings: (patch) =>
        set((s) => ({ settings: { ...s.settings, ...patch } })),

    promptHistory: [],
    addPromptMessage: (msg) =>
        set((s) => ({ promptHistory: [...s.promptHistory, msg] })),
    clearPromptHistory: () => set({ promptHistory: [] }),

    isSidebarOpen: true,
    toggleSidebar: () => set((s) => ({ isSidebarOpen: !s.isSidebarOpen })),
    isChatOpen: true,
    toggleChat: () => set((s) => ({ isChatOpen: !s.isChatOpen })),
    isSettingsOpen: false,
    toggleSettings: () => set((s) => ({ isSettingsOpen: !s.isSettingsOpen })),
    isLoading: false,
    setLoading: (isLoading) => set({ isLoading }),
}));
