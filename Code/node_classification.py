import flgo
import flgo.benchmark.partition as fbp
import flgo.algorithm.fedspa as fedspa

bmkname = 'benchmark001'
bmk_config = './config_cls.py'
bmk = flgo.gen_benchmark_from_file(bmkname, bmk_config, target_path='.', data_type='graph', task_type='node_classification')

task_config = {
    'benchmark': bmkname,
    'partitioner': {'name': fbp.NodeLouvainPartitioner, 'para': {'num_clients': 10}}
}
task = './my_louvain'
flgo.gen_task(task_config, task_path=task)

runner = flgo.init(task, fedspa, {'gpu': [0,], 'log_file': True, 'learning_rate': 0.1,
                                    'num_steps': 4, 'batch_size': 128, 'num_rounds': 200, 'proportion': 1.0, 'train_holdout': 0.4,
                                    'local_test': True, 'eval_interval': 1, 'seed': 1})

runner.run()