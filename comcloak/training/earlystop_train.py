# decay = 0.9
# patience_threshold = 5
# validation_interval = 5
# ema_val_loss = tf.Variable(initial_value=1e6, trainable=False) 
# best_val_loss = tf.Variable(initial_value=1e6, trainable=False)
# for i in range(num_training_iterations):
#     # Sampling a batch of SNRs
#     ebno_db = tf.random.uniform(shape=[batch_size_mcs], minval=ebno_db_min, maxval=ebno_db_max)
#     # Forward pass
#     with tf.GradientTape() as tape:
#         loss = model(ebno_db)
#         # Tensorflow optimizers only know how to minimize loss function.
#         # Therefore, a loss function is defined as the additive inverse of the BMD rate
#     # Computing and applying gradients
#     weights = model.trainable_weights
#     grads = tape.gradient(loss, weights)
#     optimizer.apply_gradients(zip(grads, weights))
#     # Periodically printing the progress
#     if i % validation_interval == 0:
#         print('Iteration {}/{}  Loss: {:.4f} bit'.format(i, num_training_iterations, loss.numpy()), end='\r')

#         model.validate = True
#         model.trainable = False
#         # Validation loss
#         # ebno_db = tf.linspace(val_ebno_db_min, val_ebno_db_max, num=batch_size_mcs)
#         ebno_db = tf.random.uniform(shape=[batch_size_mcs], minval=val_ebno_db_min, maxval=val_ebno_db_max)
#         val_loss = model(ebno_db)
#         print('Iteration {}/{}  Loss: {:.4f} bit  Val Loss: {:.4f} bit'.format(i, num_training_iterations, loss.numpy(), val_loss.numpy()))
#         with open(train_log_path, "a") as f:
#             # f.write(f"{i},{ebno_db:.2f},{loss.numpy():.6f}\n")
#             f.write(f"{i},{loss.numpy():.6f},{val_loss.numpy():.6f}\n")
#         ema_val_loss.assign(ema_val_loss * decay + (1 - decay) * val_loss)
#         if ema_val_loss < best_val_loss:
#             best_val_loss.assign(ema_val_loss)# not best_val_loss = ema_val_loss
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= patience_threshold:
#                 print("Stopping training due to no improvement in validation loss.")
#                 break
#         model.trainable = True

# # Save the weights in a file
# weights = model.get_weights()
# with open(model_weights_path, 'wb') as f:
#     pickle.dump(weights, f)
