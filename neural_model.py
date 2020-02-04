import nets
import numpy as np
import torch
import phyre
import logging, time
import pdb

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

class NeuralModel():

    def __init__(self):
        pass

    def train(self, cache, task_ids, tier, dev_task_ids, params):
        ###################
        ## Preprocessing ##
        ###################
        # training_data: {'task_ids', 'actions', 'simulation_statuses'}
        logging.info('Preprocess the training data')
        training_data = cache.get_sample(task_ids, params['max_train_actions'])
        # **training_data is passed as an argument to send the dictionary's keywords together

        task_indices, is_solved, actions, simulator, observations = (self._compact_simulation_data_to_trainset(tier, training_data))

        logging.info('Train set: size=%d, solved_ratio=%.2f%%', len(is_solved),is_solved.float().mean().item() * 100)

        # create the evaluation data to evaluate the training process (-> not the test data)
        logging.info('Create evaluation data from train & dev')
        eval_train = self._create_balanced_eval_set(cache, simulator.task_ids, params['eval_size'], tier)
        eval_dev   = self._create_balanced_eval_set(cache, dev_task_ids, params['eval_size'], tier)
        #eval_test = self._create_balanced_eval_set(cache, dev_task_ids, 512, tier)
        
        #################
        ## Build Model ##
        #################
        # Initialize the model
        logging.info('Start initializing the Model')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model  = self._build_model(network_type = params['network_type'],
                                   action_space_dim = simulator.action_space_dim,
                                   action_hidden_size = params['action_hidden_size'],
                                   embed_size = params['embed_size'],
                                   hidden_size = params['hidden_size'])
        # Set model to the training mode (important when using batchnorm / dropout)
        model.train()
        model.to(device)

        # Initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        if params['cosine_scheduler']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=params['updates'])
        else:
            scheduler = None

        #################
        ## Train Model ##
        #################
        rng = np.random.RandomState(42)
        # generator function to sample the batch data
        def train_indices_sampler():
            train_batch_size = params['train_batch_size']
            indices = np.arange(len(is_solved))
            # balancing is critical in the 2B-tier
            if params['balance_classes']:
                solved_mask = is_solved.numpy() > 0
                positive_indices = indices[solved_mask]
                negative_indices = indices[~solved_mask]
                half_size = train_batch_size // 2
                while True:
                    positives = rng.choice(positive_indices, size=half_size)
                    negatives = rng.choice(negative_indices, size=half_size)
                    yield np.concatenate((positives, negatives))
            else:
                while True:
                    yield rng.choice(indices, size=train_batch_size)

        logging.info('Start Training the Model')
        batch_start = 0
        losses = []
        max_loss_index = []
        max_loss = []
        mean_loss = []
        max_loss_action = []
        min_loss_index = []
        min_loss = []
        min_loss_action = []
        loss_var = []
        max_index = 0
        min_index = 0
        batch_intermediate_list = []
        last_time = time.time()
        observations = observations.to(device)
        # transfer (actions, is_solved) to the GPU's memory
        actions = actions.pin_memory()
        is_solved = is_solved.pin_memory()

        edge = self._make_edges()

        for batch_id, batch_indices in enumerate(train_indices_sampler(),start=batch_start):
            if batch_id >= params['updates']:
                break
            model.train()
            batch_task_indices = task_indices[batch_indices]
            batch_observations = observations[batch_task_indices]
            batch_actions = actions[batch_indices].to(device, non_blocking=True)
            batch_is_solved = is_solved[batch_indices].to(device, non_blocking=True)
            batch_size = batch_task_indices.shape[0]
            
            initial_featurized_objects = np.array(simulator.initial_featurized_objects)
            batch_featurized_objects = initial_featurized_objects[batch_task_indices]
            gray_list = []
            
            for i in range(batch_size):
                a = batch_featurized_objects[i].colors_one_hot[:, 4]
                gray_index = np.where(a == 1)
                gray_list.append(batch_featurized_objects[i].features[:, gray_index, :])
        
            batch_answer = np.zeros((batch_size, 16, 6, 8))
            batch_num = 0
            
            for num in batch_indices:
                simulation = simulator.simulate_action(task_indices[num].numpy(), actions[num].numpy(), need_images=True, need_featurized_objects=True)
                intermediate = np.array(simulation.featurized_objects.features)
                intermediate_full = self._make_17(intermediate)

                for t in range(16):
                    input_t = t + 1
                    
                    red_index = np.where(intermediate_full[input_t][:, 8] == 1)
                    batch_answer[batch_num][t][0] = intermediate_full[input_t][red_index[0][0]][:8]
                    green_index = np.where(intermediate_full[input_t][:, 9] == 1)
                    batch_answer[batch_num][t][1] = intermediate_full[input_t][green_index[0][0]][:8]
                    blue_index = np.where(intermediate_full[input_t][:, 10] == 1)
                    if blue_index[0].size == 0:
                        blue_index = np.where(intermediate_full[input_t][:, 11] == 1)
                    batch_answer[batch_num][t][2] = intermediate_full[input_t][blue_index[0][0]][:8]
                    gray_index = np.where(intermediate_full[input_t][:, 12] == 1)
                    for j in range(gray_index[0].size):
                        batch_answer[batch_num][t][j+3] = intermediate_full[input_t][gray_index[0][j]][:8]
                batch_num += 1

            optimizer.zero_grad()
            embedding = model(batch_observations, batch_actions, gray_list)
            qa_loss= model.compute_loss(embedding, edge, batch_answer, batch_is_solved)

            if (batch_id+1) > params['report_statistic']:
                max_loss.append(qa_loss.max().item())
                min_loss.append(qa_loss.min().item())
                mean_loss.append(qa_loss.mean().item())
                loss_var.append(qa_loss.var().item())
                max_index = qa_loss.argmax().item()
                min_index = qa_loss.argmin().item()
                max_loss_index.append(batch_task_indices[max_index])
                min_loss_index.append(batch_task_indices[min_index])
                max_loss_action.append(batch_actions[max_index])
                min_loss_action.append(batch_actions[min_index])

            loss = qa_loss# + ce_loss
            loss = torch.mean(loss)

            loss.backward()
            optimizer.step()
            #print(batch_id)
            losses.append(loss.mean().item())

            if scheduler is not None:
                scheduler.step()
            if (batch_id + 1) % params['report_every'] == 0:
                speed = params['report_every'] / (time.time() - last_time)
                last_time = time.time()
                logging.debug('Iter: %s, examples: %d, mean loss: %f, speed: %.1f batch/sec, lr: %f',
                              batch_id + 1, (batch_id + 1) * params['train_batch_size'],
                              np.mean(losses[-params['report_every']:]), speed, self._get_lr(optimizer))
            
            if (batch_id + 1) % params['eval_every'] == 0:
                logging.info('Start eval')
                eval_batch_size = params['eval_batch_size']
                stats = {}
                stats['batch_id'] = batch_id + 1
                #TODO: modify the _eval_loss
                #stats['train_loss'] = self._eval_loss(model, eval_train, eval_batch_size)
                stats['dev_loss']  = self.get_test_loss(model, cache, dev_task_ids, tier)
                #if params['num_auccess_actions'] > 0:
                #    stats['train_auccess']= self._eval_and_score_actions(cache, model, eval_train, params['num_auccess_actions'],
                #                                                         eval_batch_size, params['num_auccess_tasks'])
                #    stats['dev_auccess']  = self._eval_and_score_actions(cache, model, eval_dev, params['num_auccess_actions'],
                #                                                         eval_batch_size, params['num_auccess_tasks'])
                logging.info('__log__:%s', stats)
            #cudaFree(device)
             
            statistic = dict(max_loss = max_loss, min_loss = min_loss, loss_var= loss_var, max_loss_index = max_loss_index, min_loss_index = min_loss_index, max_loss_action= max_loss_action, min_loss_action = min_loss_action, mean_loss = mean_loss)
            
            
        return model.cpu(), statistic


    def predict_qa(self, model, cache, task_ids, tier, action):

        predicting_data = cache.get_sample(task_ids, 9)        

        edge = self._make_edges()
        task_indices, is_solved, actions, simulator, observations = (self._compact_simulation_data_to_trainset(tier, predicting_data))

        #action = actions[action_num]
        task_index = task_indices[0]
        task_index = np.array(task_index)
        action = np.array(action)
        observation = observations[0]
        observation = np.array(observation)
        observation = [observation]
        observation = np.array(observation)

        simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True)
        intermediate = np.array(simulation.images)
        intermediate = self._make_17(intermediate)
        
        action = [action]
        action = np.array(action)
        observation = torch.from_numpy(observation).float()
        action = torch.from_numpy(action).float()
        
        gray_list = []

        a = simulation.featurized_objects.colors_one_hot[:, 4]
        gray_index = np.where(a == 1)
        gray = np.array([simulation.featurized_objects.features[0, gray_index, :]])
        gray_list.append(gray)
        predict_location = model.predict_location(model(observation, action, gray_list), edge)
        predict_location = predict_location.squeeze(0).detach().cpu().numpy()
        predict_location[:, :, :4] = np.where(predict_location[:, :, :4]<0 , 0, predict_location[:, :, :4])
        predict_location[:, :, :4] = np.where(predict_location[:, :, :4]>1 , 0.99, predict_location[:, :, :4])
        for i in range(16):
            for j in range(6):
                if j == 0:
                    predict_location[i, j, 4:] = [1, 0, 0, 0]
                else:
                    predict_location[i, j, 4:] = np.where(predict_location[i, j, 4:]== predict_location[i, j, 4:].max(), 1, 0)
        color_one_hot = np.array([[1,0,0,0,0,0], 
                                  [0,1,0,0,0,0],
                                  [0,0,1,0,0,0],
                                  [0,0,0,0,1,0],
                                  [0,0,0,0,1,0],
                                  [0,0,0,0,1,0]])
        color_one_hot = np.tile(color_one_hot, (16, 1, 1))
        predict_location = np.concatenate((predict_location, color_one_hot), 2)
        obs_predict = np.zeros((16, 256, 256), dtype = int)
        for i in range(16):
            obs_predict[i, :, :] = phyre.featurized_objects_vector_to_raster(predict_location[i])
        
        return intermediate, obs_predict

    def get_test_loss(self, model, cache, task_ids, tier):
        
        eval_test = self._create_balanced_eval_set(cache, task_ids, 512, tier)
        task_indices, is_solved, actions, simulator, observations = eval_test
        losses = []
        observations = observations.to(model.device)
        batch_size = 16
        edge = self._make_edges()
        #pdb.set_trace()
        with torch.no_grad():
            model.eval()
            for i in range(0, len(task_indices), batch_size):
                batch_indices = task_indices[i:i + batch_size]
                batch_task_indices = task_indices[batch_indices]
                batch_observations = observations[batch_task_indices]
                batch_actions = actions[batch_indices]
                batch_is_solved = is_solved[batch_indices]

                initial_featurized_objects = np.array(simulator.initial_featurized_objects)
                batch_featurized_objects = initial_featurized_objects[batch_task_indices]
                gray_list = []

                for i in range(batch_size):
                    a = batch_featurized_objects[i].colors_one_hot[:, 4]
                    gray_index = np.where(a == 1)
                    gray_list.append(batch_featurized_objects[i].features[:, gray_index, :])

                batch_answer = np.zeros((batch_size, 16, 6, 8))
                batch_num = 0

                for num in batch_indices:
                    simulation = simulator.simulate_action(task_indices[num].numpy(), actions[num].numpy(), need_images=True, need_featurized_objects=True)
                    intermediate = np.array(simulation.featurized_objects.features)
                    intermediate_full = self._make_17(intermediate)


                    red_index = np.where(intermediate_full[16][:, 8] == 1)
                    batch_answer[batch_num][15][0] = intermediate_full[16][red_index[0][0]][:8]
                    green_index = np.where(intermediate_full[16][:, 9] == 1)
                    batch_answer[batch_num][15][1] = intermediate_full[16][green_index[0][0]][:8]
                    blue_index = np.where(intermediate_full[16][:, 10] == 1)
                    if blue_index[0].size == 0:
                        blue_index = np.where(intermediate_full[16][:, 11] == 1)
                    batch_answer[batch_num][15][2] = intermediate_full[16][blue_index[0][0]][:8]
                    gray_index = np.where(intermediate_full[16][:, 12] == 1)
                    for j in range(gray_index[0].size):
                        batch_answer[batch_num][15][j+3] = intermediate_full[16][gray_index[0][j]][:8]
                    batch_num += 1
                
                loss = model.compute_16_loss(model(batch_observations, batch_actions, gray_list), edge, batch_answer, batch_is_solved)
                loss = torch.mean(loss)
                losses.append(loss.mean().item())
            
        return sum(losses) / len(task_indices)
        

    def eval_actions(self, model, actions, batch_size, observation):
        """ Evaluate the score for each action with given observation """
        scores = []
        # set model to the evaluation mode
        with torch.no_grad():
            preprocessed = model.preprocess(torch.LongTensor(observation).unsqueeze(0))
            edge = self._make_edges()
            for batch_start in range(0, len(actions), batch_size):
                batch_end = min(len(actions), batch_start + batch_size)
                batch_actions = torch.FloatTensor(actions[batch_start:batch_end])
                batch_scores = model.compute_reward(model(None, batch_actions, preprocessed=preprocessed), edge)
                scores.append(batch_scores.cpu().numpy())
        return np.concatenate(scores)


    def _eval_loss(self, model, data, batch_size):
        """ Evaluate the loss for the given data & model """
        task_indices, is_solved, actions, _, observations = data
        losses = []
        observations = observations.to(model.device)
        with torch.no_grad():
            model.eval()
            for i in range(0, len(task_indices), batch_size):
                batch_indices = task_indices[i:i + batch_size]
                batch_task_indices = task_indices[batch_indices]
                batch_observations = observations[batch_task_indices]
                batch_actions = actions[batch_indices]
                batch_is_solved = is_solved[batch_indices]
                loss = model.ce_loss(model.reward_net(model(batch_observations, batch_actions)), batch_is_solved)
                losses.append(loss.mean().item())
        return sum(losses) / len(task_indices)


    def _eval_and_score_actions(self, cache, model, data, num_actions, batch_size, num_tasks):
        """ Evaluate the AUCESS for the given data & model"""
        _, _, _, simulator, observations = data

        actions = cache.action_array[:num_actions]
        indices = np.random.RandomState(1).permutation(len(observations))[:num_tasks]
        evaluator = phyre.Evaluator([simulator.task_ids[index] for index in indices])
        for i, task_index in enumerate(indices):
            scores = self.eval_actions(model, actions, batch_size, observations[task_index]).tolist()
            _, sorted_actions = zip(*sorted(zip(scores, actions), key=lambda x: (-x[0], tuple(x[1]))))
            for action in sorted_actions:
                if (evaluator.get_attempts_for_task(i) >= phyre.MAX_TEST_ATTEMPTS):
                    break
                simulation = simulator.simulate_action(task_index,
                                                      action,
                                                      need_images=False)
                evaluator.maybe_log_attempt(i, simulation.simulation_status)
        return evaluator.get_aucess()

    def _get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


    def _build_model(self, network_type, action_space_dim, action_hidden_size, embed_size, hidden_size):
        if network_type == 'resnet18':
            model = nets.ResNet18FilmAction(action_size = action_space_dim,
                                            action_hidden_size = action_hidden_size,
                                            embed_size = embed_size,
                                            hidden_size = hidden_size)
        # Our model
        elif network_type == 'resnet18_qa':
            model = nets.ResNet18PhysicalQA(action_size = action_space_dim,
                                            action_hidden_size = action_hidden_size)
        else:
            raise ValueError('Unknown network type: %s' % network_type)
        return model


    def _compact_simulation_data_to_trainset(self, tier, data):
        """
        Converts result of SimulationCache.get_data() to pytorch tensors.

        Returns a tuple (task_indices, is_solved, selected_actions, simulator, observations).
        task_indices, is_solved, selected_actions, observations are all tensors corresponding to (task, action) pair
        is_solved[i] is true iff selected_actions[i] solves task(task_ids[task_indices[i]]).
        """
        actions = data['actions']
        simulation_statuses = data['simulation_statuses']
        task_ids = data['task_ids']

        invalid = int(phyre.SimulationStatus.INVALID_INPUT)
        solved = int(phyre.SimulationStatus.SOLVED)

        # Making indices to build the (task, action) pair
        task_indices = np.repeat(np.arange(len(task_ids)).reshape((-1, 1)),
                                 actions.shape[0],
                                 axis=1).reshape(-1)
        action_indices = np.repeat(np.arange(actions.shape[0]).reshape((1, -1)),
                                   len(task_ids),
                                   axis=0).reshape(-1)
        # len(simulation_statues) = len(task) * len(action)
        simulation_statuses = simulation_statuses.reshape(-1)

        # Filter for the valid actions
        good_statuses = simulation_statuses != invalid
        is_solved = torch.LongTensor(simulation_statuses[good_statuses].astype('uint8')) == solved
        action_indices = action_indices[good_statuses]
        actions = torch.FloatTensor(actions[action_indices])
        task_indices = torch.LongTensor(task_indices[good_statuses])

        simulator = phyre.initialize_simulator(task_ids, tier)
        observations = torch.LongTensor(simulator.initial_scenes)
        #pdb.set_trace()
        return task_indices, is_solved, actions, simulator, observations


    def _create_balanced_eval_set(self, cache, task_ids, size, tier):
        """
        Prepares balanced eval set to run through a network.
        Selects (size // 2) positive (task, action) pairs and (size // 2) negative pairs and represents them into pytorch tensors.

        The format of the output is the same as in _compact_simulation_data_to_trainset.
        """
        task_ids = tuple(task_ids)
        data = cache.get_sample(task_ids)

        actions = data['actions']
        simulation_statuses = data['simulation_statuses']

        flat_statuses = simulation_statuses.reshape(-1)
        [positive_indices] = (flat_statuses == int(phyre.SimulationStatus.SOLVED)).nonzero()
        [negative_indices] = (flat_statuses == int(phyre.SimulationStatus.NOT_SOLVED)).nonzero()

        half_size = size // 2
        rng = np.random.RandomState(42)
        # If the number of indices are smaller than the half_size, indices can overlap
        positive_indices = rng.choice(positive_indices, half_size)
        negative_indices = rng.choice(negative_indices, half_size)

        all_indices = np.concatenate([positive_indices, negative_indices])
        selected_actions = torch.FloatTensor(actions[all_indices % len(actions)])
        is_solved = torch.LongTensor(flat_statuses[all_indices].astype('int')) > 0
        task_indices = torch.LongTensor(all_indices // len(actions))

        simulator = phyre.initialize_simulator(task_ids, tier)
        observations = torch.LongTensor(simulator.initial_scenes)
        return task_indices, is_solved, selected_actions, simulator, observations

    def _make_17(self, intermediate):

        original_size = intermediate.shape[0]

        if original_size != 17 :
            a = intermediate[original_size-1]
            a = np.array([a])
            #pdb.set_trace()
            for j in range(17-original_size):
                intermediate = np.concatenate((intermediate, a), axis = 0)

        return intermediate
                                               
    def _make_edges(self):
                                               
        N = np.eye(6)
        for i in range(6):
            if i == 0 :
                n = N[i]
                Rs = np.tile(n , (6,1))
            else:
                n = N[i]
                n = np.tile(n, (6,1))
                Rs = np.concatenate((Rs, n))

        Rs = np.transpose(Rs)
        M = np.eye(6)
        Rr = np.tile(M, (1,6))

        edge = dict(Rs = Rs, Rr = Rr)
                                               
        return edge
