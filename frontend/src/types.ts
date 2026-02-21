/* ─── Scene Graph Types ─── */

export interface Vec3 {
    x: number;
    y: number;
    z: number;
}

export interface WallSegment {
    id: string;
    start: Vec3;
    end: Vec3;
    height: number;
    thickness: number;
    color: string;
}

export interface FloorPlane {
    id: string;
    vertices: Vec3[];
    color: string;
    material: string;
    room_type?: string;
    room_name?: string;
}

export interface SceneObject {
    id: string;
    type: 'door' | 'window' | 'furniture' | 'fixture' | 'staircase' | 'other';
    label: string;
    position: Vec3;
    rotation: Vec3;
    scale: Vec3;
    color: string;
    geometry: 'box' | 'cylinder' | 'sphere' | 'plane';
}

export interface SceneGraph {
    walls: WallSegment[];
    floors: FloorPlane[];
    objects: SceneObject[];
    metadata: {
        name: string;
        units: 'meters' | 'feet';
        source?: string;
    };
}

/* ─── Camera Types ─── */

export type CameraView =
    | 'perspective'
    | 'isometric'
    | 'top'
    | 'front'
    | 'back'
    | 'left'
    | 'right'
    | 'walkthrough';

/* ─── Settings ─── */

export interface AppSettings {
    resolutionScale: 0.5 | 0.75 | 1 | 1.5 | 2;
    antiAliasing: boolean;
    shadows: boolean;
}

/* ─── AI Prompt ─── */

export interface PromptMessage {
    role: 'user' | 'assistant';
    content: string;
    timestamp: number;
}
