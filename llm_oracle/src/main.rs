use anchor_lang::prelude::AccountMeta;
use anchor_lang::{AccountDeserialize, AnchorSerialize, Discriminator};
use chatgpt::client::ChatGPT;
use chatgpt::config::ModelConfiguration;
use chatgpt::types::{ChatMessage, Role};
use futures::StreamExt;
use memory::InteractionMemory;
use solana_account_decoder::UiAccountEncoding;
use solana_client::pubsub_client::PubsubClient;
use solana_client::rpc_client::RpcClient;
use solana_client::rpc_config::{RpcAccountInfoConfig, RpcProgramAccountsConfig};
use solana_sdk::compute_budget::ComputeBudgetInstruction;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    instruction::Instruction,
    pubkey::Pubkey,
    signature::{Keypair, Signer},
    transaction::Transaction,
};
use std::env;
use std::error::Error;
use std::str::FromStr;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

mod memory;

const MAX_TX_RETRY_ATTEMPTS: u8 = 5;
const MAX_API_RETRY_ATTEMPTS: u8 = 3;
const OPENROUTER_API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const DEFAULT_OPENROUTER_MODEL: &str = "openrouter/free";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let (rpc_url, websocket_url, openrouter_api_key, openrouter_model, payer, identity_pda) =
        load_config();
    let openrouter_model: &'static str = Box::leak(openrouter_model.into_boxed_str());
    let mut interaction_memory = InteractionMemory::new(10);
    println!(" Oracle identity: {:?}", payer.pubkey());
    println!(" RPC: {:?}", rpc_url.as_str());
    println!(" WS: {:?}", websocket_url.as_str());
    println!(" OpenRouter model: {:?}", openrouter_model);
    loop {
        if let Err(e) = run_oracle(
            rpc_url.as_str(),
            websocket_url.as_str(),
            openrouter_api_key.as_str(),
            openrouter_model,
            &payer,
            &identity_pda,
            &mut interaction_memory,
        )
        .await
        {
            eprintln!("Error encountered: {:?}. Restarting...", e);
        }
    }
}

async fn run_oracle(
    rpc_url: &str,
    websocket_url: &str,
    openrouter_api_key: &str,
    openrouter_model: &'static str,
    payer: &Keypair,
    identity_pda: &Pubkey,
    interaction_memory: &mut InteractionMemory,
) -> Result<(), Box<dyn Error>> {
    let openrouter_client = ChatGPT::new_with_config(
        openrouter_api_key,
        ModelConfiguration {
            engine: chatgpt::config::ChatGPTEngine::Custom(openrouter_model),
            api_url: OPENROUTER_API_URL.parse()?,
            presence_penalty: 0.3,
            frequency_penalty: 0.3,
            max_tokens: Some(100),
            ..Default::default()
        },
    )?;
    let rpc_client = RpcClient::new_with_commitment(rpc_url, CommitmentConfig::processed());

    let (tx, rx) = mpsc::channel(100);
    let mut stream = ReceiverStream::new(rx);

    let rpc_config = RpcAccountInfoConfig {
        commitment: Some(CommitmentConfig::processed()),
        encoding: Some(UiAccountEncoding::Base64),
        ..Default::default()
    };

    let filters = vec![solana_client::rpc_filter::RpcFilterType::Memcmp(
        solana_client::rpc_filter::Memcmp::new(
            0,
            solana_client::rpc_filter::MemcmpEncodedBytes::Bytes(
                solana_gpt_oracle::Interaction::DISCRIMINATOR.to_vec(),
            ),
        ),
    )];

    fetch_and_process_program_accounts(
        &rpc_client,
        filters.clone(),
        payer,
        identity_pda,
        &openrouter_client,
        interaction_memory,
    )
    .await?;

    let program_config = RpcProgramAccountsConfig {
        account_config: rpc_config,
        filters: Some(filters),
        ..Default::default()
    };

    let subscription = PubsubClient::program_subscribe(
        &websocket_url,
        &solana_gpt_oracle::ID,
        Some(program_config),
    )?;

    tokio::spawn(async move {
        for update in subscription.1 {
            if tx.send(update).await.is_err() {
                eprintln!("Receiver dropped");
                break;
            }
        }
    });

    while let Some(update) = stream.next().await {
        if let Ok(interaction_pubkey) = Pubkey::from_str(&update.value.pubkey) {
            if let Some(data) = update.value.account.data.decode() {
                process_interaction(
                    payer,
                    identity_pda,
                    &openrouter_client,
                    &rpc_client,
                    interaction_pubkey,
                    data,
                    interaction_memory,
                )
                .await?;
            }
        }
    }

    Ok(())
}

/// Process an interaction and respond to it
async fn process_interaction(
    payer: &Keypair,
    identity_pda: &Pubkey,
    openrouter_client: &ChatGPT,
    rpc_client: &RpcClient,
    interaction_pubkey: Pubkey,
    data: Vec<u8>,
    interaction_memory: &mut InteractionMemory,
) -> Result<(), Box<dyn Error>> {
    if let Ok(interaction) =
        solana_gpt_oracle::Interaction::try_deserialize_unchecked(&mut data.as_slice())
    {
        if interaction.is_processed == true {
            return Ok(());
        }
        println!("Processing interaction: {:?}", interaction_pubkey);
        if let Ok(context_data) = rpc_client.get_account(&interaction.context) {
            if let Ok(context) = solana_gpt_oracle::ContextAccount::try_deserialize_unchecked(
                &mut context_data.data.as_slice(),
            ) {
                println!(
                    "Interaction: {:?}, Pubkey: {:?}",
                    interaction, interaction_pubkey
                );

                // Get a response from the OpenRouter API
                let mut previous_history = interaction_memory
                    .get_history(&interaction_pubkey)
                    .unwrap_or(Vec::new())
                    .clone();
                interaction_memory.add_interaction(
                    interaction_pubkey,
                    interaction.text.clone(),
                    Role::User,
                );
                previous_history.push(ChatMessage {
                    role: Role::User,
                    content: format!(
                        "With context: {:?}, respond to: {:?}",
                        context.text, interaction.text
                    ),
                });
                let mut api_attempts = 0;
                let mut response_content = String::new();
                while api_attempts < MAX_API_RETRY_ATTEMPTS {
                    match openrouter_client.send_history(&previous_history).await {
                        Ok(response) => {
                            response_content = response.message().content.clone();
                            break;
                        }
                        Err(e) => {
                            api_attempts += 1;
                            previous_history = previous_history
                                .iter()
                                .skip((api_attempts * 2) as usize)
                                .cloned()
                                .collect();
                            eprintln!(
                                "API call failed (attempt {}/{}): {:?}",
                                api_attempts, MAX_API_RETRY_ATTEMPTS, e
                            );
                            if api_attempts >= MAX_API_RETRY_ATTEMPTS {
                                return Err(Box::new(e));
                            }
                        }
                    }
                }

                interaction_memory.add_interaction(
                    interaction_pubkey,
                    response_content.clone(),
                    Role::System,
                );

                let response_data = [
                    solana_gpt_oracle::instruction::CallbackFromLlm::DISCRIMINATOR.to_vec(),
                    response_content.try_to_vec()?,
                ]
                .concat();

                let mut callback_instruction = Instruction {
                    program_id: solana_gpt_oracle::ID,
                    accounts: vec![
                        AccountMeta::new(payer.pubkey(), true),
                        AccountMeta::new_readonly(*identity_pda, false),
                        AccountMeta::new(interaction_pubkey, false),
                        AccountMeta::new_readonly(interaction.callback_program_id, false),
                    ],
                    data: response_data,
                };

                // Add the remaining accounts from the callback_account_metas
                let remaining_accounts: Vec<AccountMeta> = interaction
                    .callback_account_metas
                    .iter()
                    .map(|meta| AccountMeta {
                        pubkey: meta.pubkey,
                        is_signer: false,
                        is_writable: meta.is_writable,
                    })
                    .collect();
                callback_instruction.accounts.extend(remaining_accounts);

                // Send the response with the callback transaction
                let mut attempts = 0;
                while attempts < MAX_TX_RETRY_ATTEMPTS {
                    if let Ok(recent_blockhash) = rpc_client
                        .get_latest_blockhash_with_commitment(CommitmentConfig::processed())
                    {
                        let compute_budget_instruction =
                            ComputeBudgetInstruction::set_compute_unit_limit(300_000);
                        let priority_fee_instruction =
                            ComputeBudgetInstruction::set_compute_unit_price(1_000_000);

                        let transaction = Transaction::new_signed_with_payer(
                            &[
                                compute_budget_instruction,
                                priority_fee_instruction,
                                callback_instruction.clone(),
                            ],
                            Some(&payer.pubkey()),
                            &[&payer],
                            recent_blockhash.0,
                        );

                        match rpc_client.send_and_confirm_transaction(&transaction) {
                            Ok(signature) => {
                                println!("Transaction signature: {}\n", signature);
                                break;
                            }
                            Err(e) => {
                                attempts += 1;
                                eprintln!("Failed to send transaction: {:?}\n", e)
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Fetch all open interactions and process them
async fn fetch_and_process_program_accounts(
    rpc_client: &RpcClient,
    filters: Vec<solana_client::rpc_filter::RpcFilterType>,
    payer: &Keypair,
    identity_pda: &Pubkey,
    openrouter_client: &ChatGPT,
    interaction_memory: &mut InteractionMemory,
) -> Result<(), Box<dyn Error>> {
    let rpc_config = RpcAccountInfoConfig {
        commitment: Some(CommitmentConfig::processed()),
        encoding: Some(UiAccountEncoding::Base64),
        ..Default::default()
    };

    let program_config = RpcProgramAccountsConfig {
        account_config: rpc_config,
        filters: Some(filters),
        ..Default::default()
    };

    let accounts =
        rpc_client.get_program_accounts_with_config(&solana_gpt_oracle::ID, program_config)?;

    for (pubkey, account) in accounts {
        process_interaction(
            payer,
            identity_pda,
            openrouter_client,
            rpc_client,
            pubkey,
            account.data,
            interaction_memory,
        )
        .await?;
    }

    Ok(())
}

/// Load the Oracle configuration
fn load_config() -> (String, String, String, String, Keypair, Pubkey) {
    let identity = env::var("IDENTITY").unwrap_or(
        "62LxqpAW6SWhp7iKBjCQneapn1w6btAhW7xHeREWSpPzw3xZbHCfAFesSR4R76ejQXCLWrndn37cKCCLFvx6Swps"
            .to_string(),
    );
    let rpc_url = env::var("RPC_URL").unwrap_or("http://localhost:8899".to_string());
    let websocket_url = env::var("WEBSOCKET_URL").unwrap_or("ws://localhost:8900".to_string());
    let openrouter_api_key = env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set");
    let openrouter_model =
        env::var("OPENROUTER_MODEL").unwrap_or(DEFAULT_OPENROUTER_MODEL.to_string());
    let payer = Keypair::from_base58_string(&identity);
    let identity_pda = Pubkey::find_program_address(&[b"identity"], &solana_gpt_oracle::ID).0;
    (
        rpc_url,
        websocket_url,
        openrouter_api_key,
        openrouter_model,
        payer,
        identity_pda,
    )
}
