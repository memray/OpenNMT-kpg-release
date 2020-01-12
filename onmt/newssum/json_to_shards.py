
import argparse
import os


def init_opt():
    parser = argparse.ArgumentParser()
    # Input/output options
    parser.add_argument('--input_json', '-input_json', type=str, required=True, help='Path to jsonl files.')
    parser.add_argument('--output_dir', '-output_dir', default='/export/share/rmeng/data/sharded_json/')
    parser.add_argument('--output_file', '-output_file', required=True, help='such as `/cnndm/train_%d.jsonl`')
    parser.add_argument('--shard_size', '-shard_size', type=int, required=True, help='.')
    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    opt = init_opt()
    output_path = opt.output_dir + '/shard_'+str(opt.shard_size) + opt.output_file
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lines_to_write = []
    shard_count = 0
    for line in open(opt.input_json, 'r'):
        if len(lines_to_write) > 0 and len(lines_to_write) % opt.shard_size == 0:
            with open(output_path % shard_count, 'w') as output_file:
                print('Writing to %s' % (output_path % shard_count))
                for l in lines_to_write:
                    output_file.write(l)
            lines_to_write = []
            shard_count += 1
        lines_to_write.append(line)

    # flush the rest lines
    with open(output_path % shard_count, 'w') as output_file:
        print('Writing to %s' % (output_path % shard_count))
        for l in lines_to_write:
            output_file.write(l)
