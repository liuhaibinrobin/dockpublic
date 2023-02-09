
     # TEST
        print(f"=== test of epoch {epoch} ===")
        model.eval()
        with torch.no_grad():
            test_info = test_dataset.data
            test_sample_id = []
            affinity_pred_list = []
            test_result = []
            test_len = 0
            test_loss = 0
            test_recto_rate = 0
            
            
            # sample scope
            print("TEST: Compute affinity for test set.")
            for data in tqdm(test_dataloader):
                test_sample_id.extend(data.sample_id)
                data = data.to(device)
                _, affinity_pred = model(data)
                del _
                affinity_pred_list.append(affinity_pred.detach().cpu())
            # end of sample scope
            
            # write to DataFrame
            test_info["dataset_sample_id"] = test_sample_id
            test_info["affinity_pred"] = torch.cat(affinity_pred_list).tolist()
            # sample scope: ratio and loss calculate
            print("TEST: Compute loss and ratio for test set.")
            for session in tqdm(test_info.session_au.unique(), total=len(test_info.session_au.unique())):
                df = test_info[test_info.session_au==session]
                affinity_true = torch.tensor(df.value.values).to(device)
                affinity_pred = torch.tensor(df.affinity_pred.values).to(device)
                loss, recto_rate, num_pairs = pairwiseloss(pred=affinity_pred, true=affinity_true)
                loss = loss.detach().cpu()
                recto_rate = recto_rate.detach().cpu()
                session_len= len(df)
                
                test_len += np.sqrt(num_pairs)
                test_loss += loss * np.sqrt(num_pairs)
                test_recto_rate += recto_rate * np.sqrt(num_pairs)
                
                test_result.append([session, session_len, num_pairs, loss.item(), recto_rate.item()])
                # end of sample scope: ratio and loss calculate
                
            
        # save result as .csv
        test_result = pd.DataFrame(test_result, columns=["session", "length", "num_pairs", "loss", "recto_rate"])
        savepath = f"{pre}/results/test_result_{epoch}.csv"
        test_result.to_csv(savepath)
        test_info.to_csv(f"{pre}/results/test_info_{epoch}.csv")
        print(f"TEST: save test result of epoch {epoch} to {savepath}.")

        # save result to tensorboard
        test_loss /= test_len
        test_recto_rate /= test_len
        writer.add_scalar(f'rank_loss.by_epoch/test', test_loss.item(), epoch)
        writer.add_scalar(f'recto_rate.by_epoch/test', test_recto_rate.item(), epoch)
        logging.info(f"epoch {epoch} - test | rank_loss {test_loss.item()}, averaged_recto_rate {test_recto_rate.item()}")
            
        # save model checkpoint
        savepath = f"{pre}/models/epoch_{epoch}.pt"
        torch.save(model.state_dict(), savepath)
        print(f"End of epoch: save model of epoch {epoch} to {savepath}.")