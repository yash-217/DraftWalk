import { useMemo } from 'react';
import { useAppStore } from '../store/useAppStore';
import * as THREE from 'three';
import type { FloorPlane } from '../types';

/** Separate component so useMemo is called at the component level (not inside .map) */
function FloorMesh({ floor }: { floor: FloorPlane }) {
    const shape = useMemo(() => {
        const s = new THREE.Shape();
        if (floor.vertices.length < 3) return s;
        s.moveTo(floor.vertices[0].x, floor.vertices[0].z);
        for (let i = 1; i < floor.vertices.length; i++) {
            s.lineTo(floor.vertices[i].x, floor.vertices[i].z);
        }
        s.closePath();
        return s;
    }, [floor.vertices]);

    return (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]} receiveShadow>
            <shapeGeometry args={[shape]} />
            <meshStandardMaterial color={floor.color} side={THREE.DoubleSide} />
        </mesh>
    );
}

export function SceneRenderer() {
    const { scene, settings } = useAppStore();

    return (
        <group>
            <ambientLight intensity={0.6} />
            <directionalLight
                position={[15, 20, 10]}
                intensity={1.2}
                castShadow={settings.shadows}
                shadow-mapSize-width={2048}
                shadow-mapSize-height={2048}
                shadow-camera-far={60} shadow-camera-left={-20}
                shadow-camera-right={20} shadow-camera-top={20} shadow-camera-bottom={-20}
            />
            <directionalLight position={[-10, 15, -5]} intensity={0.3} />

            {/* Ground plane */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[5, -0.01, 4]} receiveShadow>
                <planeGeometry args={[40, 40]} />
                <meshStandardMaterial color="#e8e4de" />
            </mesh>

            {/* Walls */}
            {scene.walls.map((w) => {
                const dx = w.end.x - w.start.x, dz = w.end.z - w.start.z;
                const length = Math.sqrt(dx * dx + dz * dz);
                const angle = Math.atan2(dz, dx);
                return (
                    <mesh
                        key={w.id}
                        position={[(w.start.x + w.end.x) / 2, w.height / 2, (w.start.z + w.end.z) / 2]}
                        rotation={[0, -angle, 0]}
                        castShadow receiveShadow
                    >
                        <boxGeometry args={[length, w.height, w.thickness]} />
                        <meshStandardMaterial color={w.color} />
                    </mesh>
                );
            })}

            {/* Floors */}
            {scene.floors.map((f) => (
                <FloorMesh key={f.id} floor={f} />
            ))}

            {/* Objects */}
            {scene.objects.map((o) => (
                <mesh
                    key={o.id}
                    position={[o.position.x, o.position.y, o.position.z]}
                    rotation={[o.rotation.x, o.rotation.y, o.rotation.z]}
                    castShadow receiveShadow
                >
                    {o.geometry === 'cylinder' ? <cylinderGeometry args={[o.scale.x / 2, o.scale.x / 2, o.scale.y, 24]} /> :
                        o.geometry === 'sphere' ? <sphereGeometry args={[o.scale.x / 2, 24, 24]} /> :
                            o.geometry === 'plane' ? <planeGeometry args={[o.scale.x, o.scale.z]} /> :
                                <boxGeometry args={[o.scale.x, o.scale.y, o.scale.z]} />}
                    <meshStandardMaterial color={o.color} transparent={o.type === 'window'} opacity={o.type === 'window' ? 0.35 : 1} />
                </mesh>
            ))}
        </group>
    );
}
