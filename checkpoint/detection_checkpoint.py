import os
import sys
import torch
from detectron2.utils import comm
from detectron2.engine import hooks, HookBase
import logging

class PeriodicCheckpointerWithEval(HookBase):
    def __init__(self, eval_period, eval_function, checkpointer, checkpoint_period, max_to_keep=5):
        self.eval = hooks.EvalHook(eval_period, eval_function)
        self.checkpointer = hooks.PeriodicCheckpointer(checkpointer, checkpoint_period, max_to_keep=max_to_keep)
        self.best_ap = 0.0
        self.best_mr = 0.0
        self.best_r = 0.0
        self.best_sum = 0.0
        best_model_path_ap = checkpointer.save_dir + 'best_model_final_ap.pth'
        best_model_path_mr = checkpointer.save_dir + 'best_model_final_mr.pth'
        best_model_path_r = checkpointer.save_dir + 'best_model_final_r.pth'
        best_model_path_sum = checkpointer.save_dir + 'best_model_final_sum.pth'
        if os.path.isfile(best_model_path_ap):
            best_model = torch.load(best_model_path_ap, map_location=torch.device('cpu'))
            self.best_ap = best_model['AP50']
            print ("BEST AP50: ", self.best_ap)
            del best_model
        else:
            self.best_ap = 0.0
        if os.path.isfile(best_model_path_mr):
            best_model = torch.load(best_model_path_mr, map_location=torch.device('cpu'))
            self.best_mr = best_model['SGMeanRecall@100']
            print ("BEST MeanRecall@100: ", self.best_mr)
            del best_model
        else:
            self.best_mr = 0.0
        if os.path.isfile(best_model_path_r):
            best_model = torch.load(best_model_path_r, map_location=torch.device('cpu'))
            self.best_r = best_model['SGRecall@100']
            print ("BEST Recall@100: ", self.best_r)
            del best_model
        else:
            self.best_r = 0.0
        if os.path.isfile(best_model_path_sum):
            best_model = torch.load(best_model_path_sum, map_location=torch.device('cpu'))
            self.best_sum = best_model['SGRecall@100'] + best_model['SGMeanRecall@100']
            print (f"BEST pair: Recall@100-{best_model['SGRecall@100']} MeanRecall@100-{best_model['SGMeanRecall@100']}" )
            del best_model
        else:
            self.best_sum = 0.0

    def before_train(self):
        self.max_iter = self.trainer.max_iter
        self.checkpointer.max_iter = self.trainer.max_iter

    def _do_eval(self):
        results = self.eval._func()
        comm.synchronize()
        return results

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self.eval._period > 0 and next_iter % self.eval._period == 0):
            results = self._do_eval()
            if comm.is_main_process():
                try:
                    print (results)
                    dataset = 'VG_val' if 'VG_val' in results.keys() else 'VG_test'
                    current_mr = results['SG']['SGMeanRecall@100']
                    print(f'MeanRecall:{current_mr}')
                    current_r = results['SG']['SGRecall@100']
                    print(f'Recall:{current_r}')
                    current_ap = results['bbox']['AP50']
                    print(f'AP50:{current_ap}')
                    current_sum = current_r + current_mr
                    print(f'best_mr:{self.best_mr}')
                    print(f'best_r:{self.best_r}')
                    print(f'best_sum:{self.best_sum}')
                    print(f'best_ap:{self.best_ap}')
                    if current_mr > self.best_mr:
                        self.best_mr = current_mr
                        additional_state = {"iteration":self.trainer.iter, "AP50":current_ap, "SGRecall@100":current_r, "SGMeanRecall@100":current_mr}
                        self.checkpointer.checkpointer.save(
                        "best_model_final_mr", **additional_state
                        )
                    if current_r > self.best_r:
                        self.best_r = current_r
                        additional_state = {"iteration":self.trainer.iter, "AP50":current_ap, "SGRecall@100":current_r, "SGMeanRecall@100":current_mr}
                        self.checkpointer.checkpointer.save(
                        "best_model_final_r", **additional_state
                        )
                    if current_sum > self.best_sum:
                        self.best_sum = current_sum
                        additional_state = {"iteration":self.trainer.iter, "AP50":current_ap, "SGRecall@100":current_r, "SGMeanRecall@100":current_mr}
                        self.checkpointer.checkpointer.save(
                        "best_model_final_sum", **additional_state
                        ) 
                    if current_ap > self.best_ap:
                        self.best_ap = current_ap
                        additional_state = {"iteration":self.trainer.iter, "AP50":current_ap, "SGRecall@100":current_r, "SGMeanRecall@100":current_mr}
                        self.checkpointer.checkpointer.save(
                        "best_model_final_ap", **additional_state
                        )
                except:
                    current_ap = results['bbox']['AP50']
                    if current_ap > self.best_ap:
                        self.best_ap = current_ap
                        additional_state = {"iteration":self.trainer.iter, "AP50":self.best_ap}
                        self.checkpointer.checkpointer.save(
                        "best_model_final", **additional_state
                        )
        if comm.is_main_process():
            self.checkpointer.step(self.trainer.iter)
        comm.synchronize()

    def after_train(self):
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self.eval._func