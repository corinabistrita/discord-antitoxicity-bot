"""
Discord Bot pentru detectarea și moderarea toxicității
Integrat cu FastAPI backend și AI detector
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Discord.py
import discord
from discord.ext import commands, tasks

# Imports locale
from config import get_config
from ai_detector import get_ai_detector, analyze_toxicity
from database import get_database

# Setup logging
logger = logging.getLogger(__name__)

class FeedbackGenerator:
    """Generator de feedback educațional personalizat"""
    
    def __init__(self):
        self.feedback_templates = {
            'toxicity': {
                'ro': [
                    "🚨 Mesajul tău conține limbaj care poate fi ofensator. Te rog să reformulezi într-un mod mai constructiv.",
                    "⚠️ Detectez un ton agresiv în mesaj. Să încercăm o abordare mai prietenoasă?",
                    "🤝 Comunicarea respectuoasă face comunitatea mai plăcută pentru toți."
                ],
                'en': [
                    "🚨 Your message contains language that may be offensive. Please rephrase in a more constructive way.",
                    "⚠️ I detect an aggressive tone in your message. Let's try a friendlier approach?",
                    "🤝 Respectful communication makes the community more pleasant for everyone."
                ]
            },
            'hate_speech': {
                'ro': [
                    "🛑 Limbajul folosit poate fi rănitor pentru alții. Să găsim o modalitate mai bună de a te exprima.",
                    "💭 Gândește-te cum te-ai simți dacă cineva ți-ar vorbi astfel.",
                    "🌟 Diversitatea face comunitatea noastră mai puternică. Să o celebrăm!"
                ],
                'en': [
                    "🛑 The language used can be hurtful to others. Let's find a better way to express yourself.",
                    "💭 Think about how you would feel if someone spoke to you like this.",
                    "🌟 Diversity makes our community stronger. Let's celebrate it!"
                ]
            },
            'harassment': {
                'ro': [
                    "🚫 Targetarea negativă a altor persoane nu este acceptată aici.",
                    "🤝 Să încercăm să construim în loc să distrugem. Ce părere pozitivă poți împărtăși?",
                    "💡 Criticile constructive sunt binevenite, dar să fie respectuoase."
                ],
                'en': [
                    "🚫 Negative targeting of other people is not accepted here.",
                    "🤝 Let's try to build instead of destroying. What positive opinion can you share?",
                    "💡 Constructive criticism is welcome, but let it be respectful."
                ]
            },
            'profanity': {
                'ro': [
                    "🗣️ Limbajul vulgar poate face alții să se simtă inconfortabil.",
                    "✨ Vocabularul bogat arată inteligența. Să încercăm alternative creative!",
                    "🎭 Să păstrăm conversația potrivită pentru toți membrii."
                ],
                'en': [
                    "🗣️ Vulgar language can make others feel uncomfortable.",
                    "✨ Rich vocabulary shows intelligence. Let's try creative alternatives!",
                    "🎭 Let's keep the conversation appropriate for all members."
                ]
            },
            'spam': {
                'ro': [
                    "📝 Mesajele repetitive pot deranja conversația. Să păstrăm discuția relevantă!",
                    "⏰ Ia-ți timp să scrii mesaje cu sens. Calitatea contează mai mult decât cantitatea.",
                    "🎯 Să ne concentrăm pe conversații de calitate!"
                ],
                'en': [
                    "📝 Repetitive messages can disturb the conversation. Let's keep the discussion relevant!",
                    "⏰ Take time to write meaningful messages. Quality matters more than quantity.",
                    "🎯 Let's focus on quality conversations!"
                ]
            }
        }
        
        self.suggestions = {
            'toxicity': {
                'ro': "Încearcă să îți exprimi punctul de vedere fără să ataci persoana.",
                'en': "Try to express your point of view without attacking the person."
            },
            'hate_speech': {
                'ro': "Concentrează-te pe idei, nu pe caracteristici personale.",
                'en': "Focus on ideas, not personal characteristics."
            },
            'harassment': {
                'ro': "Oferă feedback constructiv în loc de critici personale.",
                'en': "Provide constructive feedback instead of personal criticism."
            },
            'profanity': {
                'ro': "Folosește un limbaj care să fie confortabil pentru toți.",
                'en': "Use language that is comfortable for everyone."
            },
            'spam': {
                'ro': "Gândește-te înainte să postezi și evită repetările.",
                'en': "Think before posting and avoid repetition."
            }
        }
    
    def generate_feedback(self, categories: List[str], score: float, 
                         user_warnings: int, language: str = 'ro') -> Dict[str, Any]:
        """Generează feedback personalizat"""
        
        primary_category = categories[0] if categories else 'toxicity'
        lang = language if language in ['ro', 'en'] else 'ro'
        
        # Selectează template-ul bazat pe numărul de avertismente
        templates = self.feedback_templates.get(primary_category, self.feedback_templates['toxicity'])
        template_list = templates.get(lang, templates['ro'])
        template_index = min(user_warnings, len(template_list) - 1)
        message = template_list[template_index]
        
        # Obține sugestia
        suggestions = self.suggestions.get(primary_category, self.suggestions['toxicity'])
        suggestion = suggestions.get(lang, suggestions['ro'])
        
        # Determină severitatea
        if score > 80:
            severity = "critical"
            emoji = "🔴"
            color = 0xff4757
        elif score > 60:
            severity = "high"
            emoji = "🟠"
            color = 0xff9500
        elif score > 40:
            severity = "medium"
            emoji = "🟡"
            color = 0xffa726
        else:
            severity = "low"
            emoji = "🟢"
            color = 0x4caf50
        
        return {
            "message": message,
            "suggestion": suggestion,
            "severity": severity,
            "emoji": emoji,
            "color": color,
            "categories": categories,
            "language": lang
        }

class AntiToxicityBot(commands.Bot):
    """Bot-ul principal anti-toxicitate"""
    
    def __init__(self):
        # Configurează intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True
        
        super().__init__(
            command_prefix=self.get_prefix,
            intents=intents,
            help_command=None
        )
        
        # Încarcă configurația
        self.config = get_config()
        
        # Componente
        self.feedback_generator = FeedbackGenerator()
        self.db = None
        self.ai_detector = None
        
        # Cache pentru performanță
        self.server_configs = {}
        self.user_warnings = {}
        
        # HTTP session pentru comunicarea cu API
        self.http_session = None
        
        # Statistici runtime
        self.stats = {
            'messages_processed': 0,
            'toxic_detected': 0,
            'actions_taken': 0,
            'start_time': datetime.utcnow()
        }
        
        print("🤖 Bot anti-toxicitate inițializat!")
    
    async def get_prefix(self, bot, message):
        """Obține prefixul pentru comenzi"""
        if not message.guild:
            return self.config.discord.command_prefix
        
        # Poți personaliza prefixul per server din cache
        server_config = self.server_configs.get(message.guild.id, {})
        return server_config.get('command_prefix', self.config.discord.command_prefix)
    
    async def setup_hook(self):
        """Setup-ul bot-ului la pornire"""
        try:
            # Inițializează database și AI detector
            self.db = await get_database()
            self.ai_detector = await get_ai_detector()
            
            # Creează HTTP session pentru API calls
            self.http_session = aiohttp.ClientSession()
            
            # Înregistrează servere existente
            await self.register_existing_guilds()
            
            # Pornește task-urile periodice
            self.update_stats.start()
            self.cleanup_cache.start()
            
            logger.info("✅ Bot setup complet!")
            
        except Exception as e:
            logger.error(f"❌ Eroare la setup bot: {e}")
            raise
    
    async def close(self):
        """Curăță resursele la închidere"""
        if self.http_session:
            await self.http_session.close()
        
        # Oprește task-urile
        if hasattr(self, 'update_stats'):
            self.update_stats.cancel()
        if hasattr(self, 'cleanup_cache'):
            self.cleanup_cache.cancel()
        
        await super().close()
    
    async def on_ready(self):
        """Eveniment când bot-ul se conectează"""
        print(f'✅ {self.user.name} este online!')
        print(f'📊 Conectat la {len(self.guilds)} servere')
        print(f'👥 Servind {sum(guild.member_count for guild in self.guilds)} utilizatori')
        
        # Setează activitatea
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name=f"toxicitatea pe {len(self.guilds)} servere 🛡️"
        )
        await self.change_presence(activity=activity)
    
    async def on_guild_join(self, guild):
        """Când bot-ul se alătură unui server nou"""
        try:
            await self.register_server(guild)
            logger.info(f"✨ Alăturat la server nou: {guild.name} ({guild.id})")
            
        except Exception as e:
            logger.error(f"Eroare la înregistrarea serverului {guild.name}: {e}")
    
    async def on_guild_remove(self, guild):
        """Când bot-ul părăsește un server"""
        # Curăță cache-ul
        self.server_configs.pop(guild.id, None)
        
        # Marchează serverul ca inactiv în baza de date
        try:
            await self.db.update_server_settings(
                guild.id, 
                {"is_active": False}
            )
            logger.info(f"👋 Părăsit server: {guild.name} ({guild.id})")
            
        except Exception as e:
            logger.error(f"Eroare la dezactivarea serverului {guild.name}: {e}")
    
    async def on_message(self, message):
        """Procesează fiecare mesaj"""
        # Ignoră mesajele de la bot-uri
        if message.author.bot:
            return
        
        # Procesează comenzile
        await self.process_commands(message)
        
        # Analizează toxicitatea doar în servere
        if message.guild:
            await self.analyze_message(message)
    
    async def analyze_message(self, message):
        """Analizează și răspunde la mesajele toxice"""
        try:
            self.stats['messages_processed'] += 1
            
            # Obține configurația serverului
            server_config = await self.get_server_config(message.guild.id)
            
            # Verifică whitelist
            if await self.is_whitelisted(message.content, message.guild.id):
                return
            
            # Analizează cu AI
            analysis = await analyze_toxicity(message.content, message.guild.id)
            
            # Verifică pragul
            threshold = server_config.get('toxicity_threshold', self.config.ai.default_threshold)
            
            if analysis.is_toxic and analysis.score >= threshold:
                await self.handle_toxic_message(message, analysis, server_config)
            
            # Trimite datele la API pentru salvare
            await self.send_to_api(message, analysis)
            
        except Exception as e:
            logger.error(f"Eroare la analiza mesajului {message.id}: {e}")
    
    async def handle_toxic_message(self, message, analysis, server_config):
        """Gestionează mesajele toxice"""
        try:
            self.stats['toxic_detected'] += 1
            
            # Obține numărul de avertismente al utilizatorului
            user_warnings = await self.get_user_warnings(message.author.id, message.guild.id)
            
            # Generează feedback
            language = analysis.language_detected or server_config.get('language', 'ro')
            feedback = self.feedback_generator.generate_feedback(
                analysis.categories, 
                analysis.score, 
                user_warnings,
                language
            )
            
            # Creează embed pentru feedback
            embed = discord.Embed(
                title=f"{feedback['emoji']} Feedback Educațional",
                description=feedback["message"],
                color=feedback["color"]
            )
            
            embed.add_field(
                name="💡 Sugestie",
                value=feedback["suggestion"],
                inline=False
            )
            
            embed.add_field(
                name="📊 Detalii Analiză",
                value=f"**Scor toxicitate:** {analysis.score:.1f}%\n"
                      f"**Categorii:** {', '.join(analysis.categories)}\n"
                      f"**Avertismente:** {user_warnings + 1}",
                inline=True
            )
            
            if analysis.bypass_detected:
                embed.add_field(
                    name="⚠️ Detecție Bypass",
                    value="Au fost detectate încercări de evitare a filtrelor",
                    inline=True
                )
            
            embed.set_footer(text="Să construim împreună o comunitate mai bună! 🌟")
            
            # Trimite feedback
            dm_sent = await self.send_feedback(message.author, embed)
            
            # Dacă nu s-a putut trimite în DM, trimite în canal
            if not dm_sent and server_config.get('public_warnings', False):
                try:
                    embed.set_footer(text=f"@{message.author.display_name}, să construim împreună o comunitate mai bună! 🌟")
                    await message.channel.send(embed=embed, delete_after=30)
                except discord.Forbidden:
                    pass
            
            # Aplică acțiuni de moderare
            await self.apply_moderation_action(message, user_warnings + 1, server_config)
            
            # Șterge mesajul dacă este configurat
            if server_config.get('auto_delete_toxic', False):
                try:
                    await message.delete()
                except discord.Forbidden:
                    pass
            
            # Actualizează avertismentele
            await self.update_user_warnings(message.author.id, message.guild.id)
            
            self.stats['actions_taken'] += 1
            
            logger.info(f"🚨 Toxic detectat: {message.author} în {message.guild.name} "
                       f"(Scor: {analysis.score:.1f}%)")
            
        except Exception as e:
            logger.error(f"Eroare la gestionarea mesajului toxic: {e}")
    
    async def send_feedback(self, user, embed):
        """Trimite feedback în DM utilizatorului"""
        try:
            await user.send(embed=embed)
            return True
        except discord.Forbidden:
            return False
        except Exception as e:
            logger.error(f"Eroare la trimiterea DM către {user}: {e}")
            return False
    
    async def apply_moderation_action(self, message, warnings, server_config):
        """Aplică acțiuni de moderare bazate pe numărul de avertismente"""
        
        escalation_steps = server_config.get('escalation_steps', self.config.moderation.escalation_steps)
        
        # Găsește acțiunea corespunzătoare
        action = None
        for step in escalation_steps:
            if warnings >= step['warnings']:
                action = step
        
        if not action:
            return
        
        try:
            if action['action'] == 'timeout' and action.get('duration', 0) > 0:
                # Timeout utilizator
                duration = timedelta(seconds=action['duration'])
                await message.author.timeout(duration, reason="Comportament toxic repetat")
                
                # Notifică în canal
                await message.channel.send(
                    f"⏰ {message.author.mention} a primit timeout {self.format_duration(duration)} "
                    f"pentru comportament toxic.",
                    delete_after=10
                )
            
            elif action['action'] == 'ban' and action.get('duration', 0) > 0:
                # Ban temporar
                await message.author.ban(
                    reason=f"Comportament toxic - {warnings} avertismente",
                    delete_message_days=0
                )
                
                # Programează unban
                asyncio.create_task(
                    self.scheduled_unban(message.guild, message.author, action['duration'])
                )
                
                await message.channel.send(
                    f"🔨 {message.author.mention} a fost banat temporar pentru comportament toxic.",
                    delete_after=10
                )
            
            # Salvează acțiunea în baza de date
            if self.db:
                await self.db.save_moderation_action(
                    server_id=await self.get_server_object_id(message.guild.id),
                    user_id=await self.get_user_object_id(message.author.id),
                    action_type=action['action'],
                    duration=action.get('duration'),
                    reason=f"Toxicitate detectată - {warnings} avertismente"
                )
            
        except discord.Forbidden:
            logger.warning(f"Nu am permisiuni pentru {action['action']} în {message.guild.name}")
        except Exception as e:
            logger.error(f"Eroare la aplicarea acțiunii {action['action']}: {e}")
    
    async def scheduled_unban(self, guild, user, duration):
        """Programează unban după o durată specificată"""
        await asyncio.sleep(duration)
        try:
            await guild.unban(user, reason="Expirare ban temporar")
            logger.info(f"Unban automat: {user} în {guild.name}")
        except Exception as e:
            logger.error(f"Eroare la unban automat: {e}")
    
    def format_duration(self, duration):
        """Formatează durata într-un string lizibil"""
        total_seconds = int(duration.total_seconds())
        
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        
        if days > 0:
            return f"{days}d {hours}h"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    async def send_to_api(self, message, analysis):
        """Trimite datele mesajului la API pentru salvare"""
        if not self.http_session:
            return
        
        try:
            data = {
                "message_id": message.id,
                "server_id": message.guild.id,
                "user_id": message.author.id,
                "channel_id": message.channel.id,
                "content": message.content,
                "analysis": analysis.dict(),
                "timestamp": message.created_at.isoformat()
            }
            
            headers = {
                "Authorization": f"Bearer {self.config.fastapi.secret_key}",
                "Content-Type": "application/json"
            }
            
            # Trimite la webhook-ul API
            api_url = f"http://localhost:{self.config.fastapi.port}/api/webhook/discord/message"
            
            async with self.http_session.post(api_url, json=data, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"API webhook failed: {response.status}")
            
        except Exception as e:
            logger.error(f"Eroare la trimiterea către API: {e}")
    
    # === CACHE MANAGEMENT ===
    
    async def get_server_config(self, guild_id):
        """Obține configurația serverului din cache sau baza de date"""
        if guild_id in self.server_configs:
            return self.server_configs[guild_id]
        
        try:
            if self.db:
                server = await self.db.get_server_by_discord_id(guild_id)
                config = server.settings if server else {}
            else:
                config = {}
            
            # Aplică valorile default
            default_config = self.config.get_server_config(guild_id)
            final_config = {**default_config, **config}
            
            # Cache configurația
            self.server_configs[guild_id] = final_config
            return final_config
            
        except Exception as e:
            logger.error(f"Eroare la obținerea config server {guild_id}: {e}")
            return self.config.get_server_config(guild_id)
    
    async def get_user_warnings(self, user_id, guild_id):
        """Obține numărul de avertismente al utilizatorului"""
        cache_key = f"{guild_id}:{user_id}"
        
        if cache_key in self.user_warnings:
            return self.user_warnings[cache_key]
        
        try:
            if self.db:
                user = await self.db.get_user_by_discord_id(user_id)
                warnings = user.warnings if user else 0
            else:
                warnings = 0
            
            self.user_warnings[cache_key] = warnings
            return warnings
            
        except Exception as e:
            logger.error(f"Eroare la obținerea warnings user {user_id}: {e}")
            return 0
    
    async def update_user_warnings(self, user_id, guild_id):
        """Actualizează avertismentele utilizatorului"""
        try:
            if self.db:
                new_warnings = await self.db.add_user_warning(user_id)
                
                # Actualizează cache-ul
                cache_key = f"{guild_id}:{user_id}"
                self.user_warnings[cache_key] = new_warnings
                
                return new_warnings
        except Exception as e:
            logger.error(f"Eroare la actualizarea warnings user {user_id}: {e}")
            return 0
    
    async def is_whitelisted(self, content, guild_id):
        """Verifică dacă mesajul este în whitelist"""
        try:
            if self.db:
                server = await self.db.get_server_by_discord_id(guild_id)
                if server:
                    whitelist = await self.db.get_whitelist(server.id)
                    
                    for entry in whitelist:
                        if entry.is_regex:
                            import re
                            if re.search(entry.pattern, content, re.IGNORECASE):
                                return True
                        else:
                            if entry.pattern.lower() in content.lower():
                                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Eroare la verificarea whitelist: {e}")
            return False
    
    async def register_existing_guilds(self):
        """Înregistrează toate serverele existente"""
        for guild in self.guilds:
            await self.register_server(guild)
    
    async def register_server(self, guild):
        """Înregistrează un server în baza de date"""
        try:
            if self.db:
                await self.db.create_or_update_server(
                    discord_id=guild.id,
                    name=guild.name,
                    settings={"is_active": True}
                )
        except Exception as e:
            logger.error(f"Eroare la înregistrarea serverului {guild.name}: {e}")
    
    async def get_server_object_id(self, discord_id):
        """Obține ObjectId pentru server din Discord ID"""
        try:
            server = await self.db.get_server_by_discord_id(discord_id)
            return server.id if server else None
        except:
            return None
    
    async def get_user_object_id(self, discord_id):
        """Obține ObjectId pentru user din Discord ID"""
        try:
            user = await self.db.get_user_by_discord_id(discord_id)
            return user.id if user else None
        except:
            return None
    
    # === TASK-URI PERIODICE ===
    
    @tasks.loop(minutes=30)
    async def update_stats(self):
        """Actualizează statisticile periodice"""
        try:
            # Actualizează activitatea bot-ului
            activity = discord.Activity(
                type=discord.ActivityType.watching,
                name=f"toxicitatea pe {len(self.guilds)} servere | "
                     f"{self.stats['toxic_detected']} detectate 🛡️"
            )
            await self.change_presence(activity=activity)
            
            # Log statistici
            uptime = datetime.utcnow() - self.stats['start_time']
            logger.info(f"📊 Stats: {self.stats['messages_processed']} mesaje, "
                       f"{self.stats['toxic_detected']} toxice, "
                       f"uptime: {self.format_duration(uptime)}")
            
        except Exception as e:
            logger.error(f"Eroare la actualizarea stats: {e}")
    
    @tasks.loop(hours=1)
    async def cleanup_cache(self):
        """Curăță cache-ul periodic"""
        try:
            # Limitează dimensiunea cache-ului
            if len(self.server_configs) > 1000:
                # Păstrează doar primele 500
                items = list(self.server_configs.items())[:500]
                self.server_configs = dict(items)
            
            if len(self.user_warnings) > 10000:
                # Păstrează doar primele 5000
                items = list(self.user_warnings.items())[:5000]
                self.user_warnings = dict(items)
            
            logger.debug("🧹 Cache cleanup completat")
            
        except Exception as e:
            logger.error(f"Eroare la cleanup cache: {e}")

# === COMENZI BOT ===

@commands.command(name="toxicity-check")
@commands.has_permissions(manage_messages=True)
async def toxicity_check(ctx, *, message_text):
    """Testează manual un mesaj pentru toxicitate"""
    try:
        analysis = await analyze_toxicity(message_text)
        
        embed = discord.Embed(
            title="🔍 Analiză Toxicitate",
            description=f"**Mesaj:** {message_text[:100]}{'...' if len(message_text) > 100 else ''}",
            color=0xff6b6b if analysis.is_toxic else 0x4caf50
        )
        
        embed.add_field(
            name="📊 Rezultat",
            value=f"**Scor:** {analysis.score:.1f}%\n"
                  f"**Toxic:** {'Da' if analysis.is_toxic else 'Nu'}\n"
                  f"**Încredere:** {analysis.confidence:.2f}\n"
                  f"**Model:** {analysis.model_used}",
            inline=True
        )
        
        if analysis.categories:
            embed.add_field(
                name="🏷️ Categorii",
                value=', '.join(analysis.categories),
                inline=True
            )
        
        if analysis.bypass_detected:
            embed.add_field(
                name="⚠️ Bypass",
                value="Detectat",
                inline=True
            )
        
        embed.add_field(
            name="⏱️ Timp Procesare",
            value=f"{analysis.processing_time:.3f}s",
            inline=True
        )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Eroare la analiza mesajului: {e}")

@commands.command(name="set-threshold")
@commands.has_permissions(administrator=True)
async def set_threshold(ctx, threshold: float):
    """Setează pragul de sensibilitate (0-100)"""
    if not 0 <= threshold <= 100:
        await ctx.send("❌ Pragul trebuie să fie între 0 și 100")
        return
    
    try:
        # Actualizează în baza de date
        if ctx.bot.db:
            await ctx.bot.db.update_server_settings(
                ctx.guild.id,
                {"toxicity_threshold": threshold}
            )
        
        # Curăță cache-ul pentru server
        ctx.bot.server_configs.pop(ctx.guild.id, None)
        
        await ctx.send(f"✅ Pragul de detectare setat la {threshold}%")
        
    except Exception as e:
        await ctx.send(f"❌ Eroare la setarea pragului: {e}")

@commands.command(name="user-stats")
@commands.has_permissions(manage_messages=True)
async def user_stats(ctx, user: discord.Member):
    """Afișează statisticile unui utilizator"""
    try:
        if not ctx.bot.db:
            await ctx.send("❌ Baza de date nu este disponibilă")
            return
        
        # Obține datele utilizatorului
        user_data = await ctx.bot.db.get_user_by_discord_id(user.id)
        
        if not user_data:
            await ctx.send(f"❌ Nu există date pentru {user.display_name}")
            return
        
        embed = discord.Embed(
            title=f"📊 Statistici - {user.display_name}",
            color=0x36393f
        )
        
        embed.set_thumbnail(url=user.display_avatar.url)
        
        embed.add_field(
            name="📈 Activitate",
            value=f"**Total mesaje:** {user_data.total_messages:,}\n"
                  f"**Mesaje toxice:** {user_data.toxic_messages:,}\n"
                  f"**Avertismente:** {user_data.warnings}\n"
                  f"**Rata toxicitate:** {user_data.toxicity_ratio:.2f}%",
            inline=True
        )
        
        # Status
        if user_data.toxicity_ratio > 10:
            status = "🔴 Risc Ridicat"
            color = 0xff4757
        elif user_data.toxicity_ratio > 5:
            status = "🟡 Atenție"
            color = 0xffa726
        else:
            status = "🟢 Normal"
            color = 0x4caf50
        
        embed.color = color
        
        embed.add_field(
            name="⚠️ Status",
            value=status,
            inline=True
        )
        
        if user_data.last_activity:
            embed.add_field(
                name="🕒 Ultima Activitate",
                value=f"<t:{int(user_data.last_activity.timestamp())}:R>",
                inline=True
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Eroare la obținerea statisticilor: {e}")

@commands.command(name="bot-stats")
async def bot_stats(ctx):
    """Afișează statisticile bot-ului"""
    
    embed = discord.Embed(
        title="📊 Statistici Bot Anti-Toxicitate",
        color=0x5865f2
    )
    
    embed.set_thumbnail(url=ctx.bot.user.display_avatar.url)
    
    # Statistici generale
    uptime = datetime.utcnow() - ctx.bot.stats['start_time']
    
    embed.add_field(
        name="🤖 General",
        value=f"**Servere:** {len(ctx.bot.guilds):,}\n"
              f"**Utilizatori:** {sum(guild.member_count for guild in ctx.bot.guilds):,}\n"
              f"**Uptime:** {ctx.bot.format_duration(uptime)}",
        inline=True
    )
    
    embed.add_field(
        name="📈 Activitate",
        value=f"**Mesaje procesate:** {ctx.bot.stats['messages_processed']:,}\n"
              f"**Toxicitate detectată:** {ctx.bot.stats['toxic_detected']:,}\n"
              f"**Acțiuni aplicate:** {ctx.bot.stats['actions_taken']:,}",
        inline=True
    )
    
    # Performanță
    if ctx.bot.stats['messages_processed'] > 0:
        detection_rate = (ctx.bot.stats['toxic_detected'] / ctx.bot.stats['messages_processed']) * 100
        embed.add_field(
            name="⚡ Performanță",
            value=f"**Rata detectare:** {detection_rate:.2f}%\n"
                  f"**Cache servere:** {len(ctx.bot.server_configs)}\n"
                  f"**Cache utilizatori:** {len(ctx.bot.user_warnings)}",
            inline=True
        )
    
    # AI Stats
    if ctx.bot.ai_detector:
        ai_stats = ctx.bot.ai_detector.get_stats()
        embed.add_field(
            name="🧠 AI Detector",
            value=f"**Model:** {ai_stats['config']['model_name']}\n"
                  f"**Cache hit rate:** {ai_stats['cache_hit_rate_percent']:.1f}%\n"
                  f"**Timp mediu:** {ai_stats['average_processing_time']:.3f}s",
            inline=True
        )
    
    embed.set_footer(text=f"Bot v{ctx.bot.config.version} | Timp răspuns: {ctx.bot.latency*1000:.1f}ms")
    
    await ctx.send(embed=embed)

# Adaugă comenzile la bot
def setup_commands(bot):
    """Adaugă comenzile la bot"""
    bot.add_command(toxicity_check)
    bot.add_command(set_threshold)
    bot.add_command(user_stats)
    bot.add_command(bot_stats)

# === FUNCȚIA PRINCIPALĂ ===

async def main():
    """Funcția principală pentru rularea bot-ului"""
    
    # Inițializează configurația
    config = get_config()
    
    # Validează token-ul
    if not config.discord.bot_token:
        logger.error("❌ DISCORD_BOT_TOKEN nu este setat!")
        return
    
    # Creează bot-ul
    bot = AntiToxicityBot()
    
    # Adaugă comenzile
    setup_commands(bot)
    
    try:
        # Pornește bot-ul
        await bot.start(config.discord.bot_token)
        
    except discord.LoginFailure:
        logger.error("❌ Token Discord invalid!")
    except Exception as e:
        logger.error(f"❌ Eroare la pornirea bot-ului: {e}")
    finally:
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main())