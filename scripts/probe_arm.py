"""Read SO101 motor feedback without configuring motors or changing torque."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True)
    args = parser.parse_args()
    # The motor SDK is enough for diagnostics; importing LeRobot also loads torch.
    import rustypot
    joints = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
    model, definition = rustypot.find_model('sts3215')
    registers = ('present_position', 'goal_position', 'torque_enable', 'operating_mode',
                 'min_position_limit', 'max_position_limit', 'homing_offset',
                 'present_load', 'present_voltage', 'present_temperature')
    result = {'port': args.port, 'motors': {}, 'errors': {}}
    # rustypot's context manager closes the serial port without changing torque.
    with rustypot.Bus(args.port, 1_000_000, 0.1, {i: definition for i in range(1, 7)}) as bus:
        for motor_id, joint in enumerate(joints, 1):
            values = {'id': motor_id}
            result['motors'][joint] = values
            try:
                for register in ('model_number', *registers):
                    value, error = bus.read_register_with_error(motor_id, register, retries=1)
                    if error:
                        raise RuntimeError(f'{register}: motor status {error}')
                    values[register] = value
                    if register == 'model_number' and value != model:
                        raise RuntimeError(f'Expected STS3215 model {model}, found {value}')
            except (OSError, RuntimeError) as exc:
                result['errors'][joint] = str(exc)
    print(json.dumps(result, indent=2))
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
