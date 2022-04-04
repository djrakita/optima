use crate::robot_modules::robot_configuration_generator_module::RobotConfigurationGeneratorModule;
use crate::robot_modules::robot_configuration_module::RobotConfigurationModule;
use crate::utils::utils_errors::OptimaError;

#[derive(Clone, Debug)]
pub struct RobotDOFModule {
    num_dofs: usize
}
impl RobotDOFModule {
    pub fn new(robot_name: &str, configuration_name: Option<&str>) -> Result<Self, OptimaError> {
        let robot_configuration = RobotConfigurationGeneratorModule::new(robot_name)?.generate_configuration(configuration_name)?;

        let mut out_self = Self {
            num_dofs: 0
        };

        out_self.set_num_dofs(&robot_configuration);

        return Ok(out_self);
    }

    fn set_num_dofs(&mut self, robot_configuration: &RobotConfigurationModule) {
        let mut num_dofs = 0 as usize;
        let joints = robot_configuration.robot_model_module().joints();

        for j in joints {
            if j.active() {
                num_dofs += j.num_dofs();
            }
        }

        self.num_dofs = num_dofs;
    }

    pub fn num_dofs(&self) -> usize {
        self.num_dofs
    }
}