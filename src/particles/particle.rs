use crate::math::Vec3;

#[derive(Debug, Clone)]
pub struct Particle {
    pub position: Vec3,
    pub direction: Vec3,
    pub rotation: f32,
    death_time: f32,
    life_time: f32,
}

impl Particle {
    pub fn new() -> Self {
        Self {
            position: Vec3::new(640.0 * (0.5 - rand::random::<f32>()), -240.0, 0.0),
            direction: Vec3::new(640.0 * (0.5 - rand::random::<f32>()), -240.0, 0.0),
            rotation: rand::random::<f32>(),
            death_time: rand::random::<f32>(),
            life_time: 0.0,
        }
    }

    pub fn update(&mut self, new_position: Vec3, delta_time: &f32) {
        self.position = self.position.add(&new_position);
        self.rotation += 0.01;
        self.life_time += delta_time;
        if self.life_time > self.death_time {
            *self = Self::new();
        }
    }
}
