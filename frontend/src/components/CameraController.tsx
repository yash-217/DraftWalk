import { useEffect, useRef, useCallback } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useAppStore } from '../store/useAppStore';
import * as THREE from 'three';
import type { CameraView } from '../types';

const ORTHO_DISTANCE = 20;
const SCENE_CENTER = new THREE.Vector3(5, 1, 4);

function getCameraConfig(view: CameraView) {
    switch (view) {
        case 'isometric': return { pos: [15, 15, 15] as const, isOrtho: true };
        case 'top': return { pos: [5, ORTHO_DISTANCE, 4] as const, isOrtho: true };
        case 'front': return { pos: [5, 3, -ORTHO_DISTANCE] as const, isOrtho: true };
        case 'back': return { pos: [5, 3, ORTHO_DISTANCE + 8] as const, isOrtho: true };
        case 'left': return { pos: [-ORTHO_DISTANCE, 3, 4] as const, isOrtho: true };
        case 'right': return { pos: [ORTHO_DISTANCE + 10, 3, 4] as const, isOrtho: true };
        default: return { pos: [12, 8, 12] as const, isOrtho: false };
    }
}

export function CameraController() {
    const cameraView = useAppStore((s) => s.cameraView);
    const { camera } = useThree();

    useEffect(() => {
        if (cameraView === 'walkthrough') return;
        const config = getCameraConfig(cameraView);
        camera.position.set(...config.pos);
        camera.lookAt(SCENE_CENTER);
        camera.updateProjectionMatrix();
    }, [cameraView, camera]);

    if (cameraView === 'walkthrough') return null;
    return <OrbitControls target={SCENE_CENTER} enableDamping dampingFactor={0.08} maxPolarAngle={Math.PI / 2} />;
}

export function WalkthroughController() {
    const cameraView = useAppStore((s) => s.cameraView);
    const { camera, gl } = useThree();

    const keys = useRef<Record<string, boolean>>({});
    const euler = useRef(new THREE.Euler(0, 0, 0, 'YXZ'));
    const isLocked = useRef(false);

    const SPEED = 4;
    const MOUSE_SENSITIVITY = 0.002;
    const EYE_HEIGHT = 1.6;

    const onKeyDown = useCallback((e: KeyboardEvent) => { keys.current[e.code] = true; }, []);
    const onKeyUp = useCallback((e: KeyboardEvent) => { keys.current[e.code] = false; }, []);

    const onMouseMove = useCallback((e: MouseEvent) => {
        if (!isLocked.current) return;
        euler.current.setFromQuaternion(camera.quaternion);
        euler.current.y -= e.movementX * MOUSE_SENSITIVITY;
        euler.current.x -= e.movementY * MOUSE_SENSITIVITY;
        euler.current.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, euler.current.x));
        camera.quaternion.setFromEuler(euler.current);
    }, [camera]);

    const onPointerLockChange = useCallback(() => {
        isLocked.current = document.pointerLockElement === gl.domElement;
    }, [gl]);

    useEffect(() => {
        if (cameraView !== 'walkthrough') {
            if (document.pointerLockElement === gl.domElement) document.exitPointerLock();
            return;
        }

        camera.position.set(5, EYE_HEIGHT, 4);
        camera.lookAt(8, EYE_HEIGHT, 4);

        const requestLock = () => gl.domElement.requestPointerLock();
        gl.domElement.addEventListener('click', requestLock);
        document.addEventListener('pointerlockchange', onPointerLockChange);
        document.addEventListener('mousemove', onMouseMove);
        window.addEventListener('keydown', onKeyDown);
        window.addEventListener('keyup', onKeyUp);

        return () => {
            gl.domElement.removeEventListener('click', requestLock);
            document.removeEventListener('pointerlockchange', onPointerLockChange);
            document.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('keydown', onKeyDown);
            window.removeEventListener('keyup', onKeyUp);
            if (document.pointerLockElement === gl.domElement) document.exitPointerLock();
        };
    }, [cameraView, camera, gl, onKeyDown, onKeyUp, onMouseMove, onPointerLockChange]);

    useFrame((_, delta) => {
        if (cameraView !== 'walkthrough') return;
        const move = new THREE.Vector3();
        if (keys.current['KeyW'] || keys.current['ArrowUp']) move.z -= 1;
        if (keys.current['KeyS'] || keys.current['ArrowDown']) move.z += 1;
        if (keys.current['KeyA'] || keys.current['ArrowLeft']) move.x -= 1;
        if (keys.current['KeyD'] || keys.current['ArrowRight']) move.x += 1;

        if (move.lengthSq() > 0) {
            move.normalize().multiplyScalar(SPEED * delta);
            move.applyQuaternion(camera.quaternion);
            camera.position.add(move);
            camera.position.y = EYE_HEIGHT;
        }
    });

    return null;
}
