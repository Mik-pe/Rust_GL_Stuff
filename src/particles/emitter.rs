use super::Particle;
use crate::math::Vec3;

#[derive(Debug)]
pub struct Emitter {
    pub particle_list: Vec<Particle>,
    origin: Vec3,
}

impl Emitter {
    pub fn new(num_particles: i32) -> Emitter {
        let mut parts = Vec::new();
        for _i in 0..num_particles {
            parts.push(Particle::new());
        }

        Emitter {
            particle_list: parts,
            origin: Vec3::new(0.0, 0.0, 0.0),
        }
    }

    pub fn tick(&mut self, delta_time: f32) {
        for particle in &mut self.particle_list {
            particle.update(Vec3::new(-0.003, 0.0, 0.0), &delta_time);
        }
    }
}
