import pandas as pd
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer, util
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from multiprocessing import Pool

# 更丰富的同义词/技能词库
SYNONYM_MAP = {
    "python": ["python", "python3", "python开发", "py"],
    "java": ["java", "java开发", "jvm"],
    "c++": ["c++", "cpp", "c plus plus"],
    "c#": ["c#", "c sharp", "csharp"],
    "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript", "ts"],
    "sql": ["sql", "数据库", "structured query language", "mysql", "postgresql", "oracle", "sqlite"],
    "excel": ["excel", "microsoft excel", "表格"],
    "linux": ["linux", "unix", "ubuntu", "centos", "redhat"],
    "git": ["git", "github", "gitlab", "版本控制"],
    "docker": ["docker", "容器", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "machine learning": ["machine learning", "ml", "机器学习", "sklearn", "scikit-learn"],
    "deep learning": ["deep learning", "dl", "深度学习", "tensorflow", "pytorch", "keras"],
    "nlp": ["nlp", "自然语言处理", "bert", "transformer", "text mining"],
    "data analysis": ["data analysis", "数据分析", "data analytics", "pandas", "numpy"],
    "data visualization": ["data visualization", "数据可视化", "matplotlib", "seaborn", "plotly"],
    "project management": ["project management", "项目管理", "pmp", "scrum", "agile"],
    "communication": ["communication", "沟通", "presentation", "演讲", "public speaking"],
    "teamwork": ["teamwork", "团队合作", "collaboration"],
    "leadership": ["leadership", "领导力", "team lead", "manager"],
    "sales": ["sales", "销售", "business development", "bd"],
    "marketing": ["marketing", "市场", "digital marketing", "seo", "sem"],
    "finance": ["finance", "金融", "accounting", "会计", "cpa"],
    "customer service": ["customer service", "客服", "support"],
    "problem solving": ["problem solving", "解决问题", "troubleshooting"],
    "time management": ["time management", "时间管理"],
    "english": ["english", "英语"],
    "chinese": ["chinese", "中文", "mandarin"],
    "japanese": ["japanese", "日语"],
    "r": ["r", "r语言"],
    "go": ["go", "golang"],
    "php": ["php"],
    "html": ["html", "hypertext markup language"],
    "css": ["css", "cascading style sheets"],
    "react": ["react", "reactjs", "react.js"],
    "vue": ["vue", "vuejs", "vue.js"],
    "angular": ["angular", "angularjs", "angular.js"],
    "swift": ["swift", "ios开发"],
    "android": ["android", "安卓开发"],
    "objective-c": ["objective-c", "objc"],
    "shell": ["shell", "bash", "sh", "命令行"],
    "network": ["network", "网络", "tcp/ip", "tcp", "udp"],
    "rest api": ["rest api", "restful", "api", "web api"],
    "cloud": ["cloud", "aws", "azure", "gcp", "云计算"],
}

def normalize_skill(skill):
    if not isinstance(skill, str):
        if pd.isna(skill):
            return ""
        skill = str(skill)
    skill = skill.lower().strip()
    for norm, variants in SYNONYM_MAP.items():
        if skill in [v.lower() for v in variants]:
            return norm
    return skill

def normalize_skill_set(skills):
    return set(normalize_skill(s) for s in skills)

class ResumeParser:
    SUPPORTED_FORMATS = {'.pdf', '.docx'}

    @staticmethod
    def load_resume(file_path: str) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Resume file not found: {file_path}")
        if file_path.suffix.lower() not in ResumeParser.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format. Supported formats: {ResumeParser.SUPPORTED_FORMATS}")
        if file_path.suffix.lower() == '.pdf':
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        else:
            from docx import Document
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    @staticmethod
    def extract_name(resume_text: str, file_path: str = None) -> str:
        if file_path:
            filename = Path(file_path).stem
            name = filename.replace('_', ' ').replace('-', ' ')
            if ' ' in name and 5 <= len(name) <= 40:
                return ' '.join(part.capitalize() for part in name.split())
        lines = resume_text.strip().split('\n')
        for line in lines[:10]:
            if 3 <= len(line.strip()) <= 40 and ' ' in line and not any(word in line.lower() for word in ['resume', 'cv']):
                return line.strip()
        return "Candidate"

    @staticmethod
    def generate_resume_summary(resume_text: str) -> dict:
        summary = {
            "skills": [],
            "education": [],
            "experience": [],
        }
        sections = {}
        current_section = "header"
        section_text = []
        for line in resume_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if re.match(r'^(EXPERIENCE|EDUCATION|SKILLS)', line.upper()):
                if section_text:
                    sections[current_section] = '\n'.join(section_text)
                    section_text = []
                current_section = line.lower()
            else:
                section_text.append(line)
        if section_text:
            sections[current_section] = '\n'.join(section_text)
        # 技能
        if "skills" in sections:
            skill_text = sections["skills"]
            skills = re.findall(r'[\w\+\#\.]+', skill_text)
            summary["skills"] = [s for s in skills if len(s) > 2][:15]
        # 学历
        if "education" in sections:
            edu_text = sections["education"]
            degrees = re.findall(r'(Bachelor|Master|PhD|B\.S\.|M\.S\.|MBA|BA|BS|MA|MS|MD|JD|MSc|MEng)', edu_text, re.I)
            summary["education"] = list(set(degrees))
        # 经验
        if "experience" in sections:
            exp_text = sections["experience"]
            jobs = re.findall(r'([A-Z][A-Za-z\\s]+) at ([A-Za-z\\s]+)', exp_text)
            summary["experience"] = [f"{title} at {company}" for title, company in jobs][:3]
        return summary

class JobMatcher:
    def __init__(self):
        # 多模型加载
        self.model_mpnet = SentenceTransformer('all-mpnet-base-v2')
        self.model_minilm = SentenceTransformer('all-MiniLM-L6-v2')
        self.model_multi = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.stopwords = set([
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when', 'where', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'to', 'of', 'for', 'with',
            'in', 'on', 'at', 'by', 'from', 'up', 'about', 'into', 'over', 'after', 'beneath', 'under', 'above', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'who', 'which', 'your', 'our',
            'their', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'would', 'could', 'should', 'shall', 'might', 'must', 'there', 'etc', 'also', 'him', 'her', 'them'
        ])

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            if pd.isna(text):
                return ""
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def compute_similarity(self, jobs_df: pd.DataFrame, resume_text: str) -> dict:
        job_texts = jobs_df['description'].fillna('').apply(self.preprocess_text).tolist()
        # 三个模型分别编码
        emb_mpnet = self.model_mpnet.encode([resume_text], convert_to_tensor=True)
        emb_minilm = self.model_minilm.encode([resume_text], convert_to_tensor=True)
        emb_multi = self.model_multi.encode([resume_text], convert_to_tensor=True)
        job_vecs_mpnet = self.model_mpnet.encode(job_texts, convert_to_tensor=True, show_progress_bar=True)
        job_vecs_minilm = self.model_minilm.encode(job_texts, convert_to_tensor=True, show_progress_bar=True)
        job_vecs_multi = self.model_multi.encode(job_texts, convert_to_tensor=True, show_progress_bar=True)
        # 计算相似度
        scores_mpnet = util.cos_sim(emb_mpnet, job_vecs_mpnet)[0].cpu().numpy()
        scores_minilm = util.cos_sim(emb_minilm, job_vecs_minilm)[0].cpu().numpy()
        scores_multi = util.cos_sim(emb_multi, job_vecs_multi)[0].cpu().numpy()
        return {
            "mpnet": scores_mpnet,
            "minilm": scores_minilm,
            "multi": scores_multi
        }

    def extract_keywords(self, text: str, top_n: int = 20) -> list:
        text = self.preprocess_text(text)
        words = text.split()
        filtered = [w for w in words if w not in self.stopwords and len(w) > 2]
        freq = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    def extract_key_phrases(self, text: str, top_n: int = 10) -> list:
        patterns = [
            r'(?:must have|required)\s+([A-Za-z0-9\s]{5,50}?)(?:[.,;:]|$)',
            r'(?:experience (?:with|in))\s+([A-Za-z0-9\s]{5,50}?)(?:[.,;:]|$)',
            r'(?:knowledge of)\s+([A-Za-z0-9\s]{5,50}?)(?:[.,;:]|$)',
        ]
        phrases = []
        for pat in patterns:
            phrases += [m.strip() for m in re.findall(pat, text, re.I) if len(m.strip()) > 5]
        return list(dict.fromkeys(phrases))[:top_n]

    def analyze_match(self, resume_text: str, job_text: str, resume_skills=None, job_skills=None) -> dict:
        resume_keywords = self.extract_keywords(resume_text, 30)
        job_keywords = self.extract_keywords(job_text, 30)
        job_phrases = self.extract_key_phrases(job_text, 10)
        # 技能同义词归一化
        if resume_skills is not None and job_skills is not None:
            resume_skills_set = normalize_skill_set(resume_skills)
            job_skills_set = normalize_skill_set(job_skills)
            skill_common = resume_skills_set & job_skills_set
            skill_missing = job_skills_set - resume_skills_set
            skill_ratio = len(skill_common) / max(1, len(job_skills_set))
        else:
            skill_common, skill_missing, skill_ratio = [], [], 0
        # 关键词交集
        common = [w for w in resume_keywords if w in job_keywords]
        missing = [w for w in job_keywords if w not in resume_keywords]
        ratio = len(common) / len(job_keywords) if job_keywords else 0
        return {
            "common_keywords": common[:10],
            "missing_keywords": missing[:10],
            "job_key_phrases": job_phrases,
            "keyword_match_ratio": ratio,
            "skill_common": list(skill_common) if resume_skills is not None else [],
            "skill_missing": list(skill_missing) if resume_skills is not None else [],
            "skill_match_ratio": skill_ratio
        }

    def batch_process_jobs(self, jobs_df: pd.DataFrame, batch_size=32):
        """批量处理岗位"""
        results = []
        for i in range(0, len(jobs_df), batch_size):
            batch = jobs_df.iloc[i:i+batch_size]
            batch_results = self.process_batch(batch)
            results.extend(batch_results)
        return results

    @staticmethod
    def parallel_process(func, data, n_processes=4):
        """并行处理"""
        with Pool(n_processes) as pool:
            results = pool.map(func, data)
        return results

class EnhancedJobMatcher(JobMatcher):
    def calculate_experience_match(self, resume_exp: list, job_req: str) -> float:
        """计算经验匹配度"""
        # 实现经验匹配逻辑
        pass

    def calculate_education_match(self, resume_edu: list, job_req: str) -> float:
        """计算学历匹配度"""
        # 实现学历匹配逻辑
        pass

    def calculate_location_match(self, resume_loc: str, job_loc: str) -> float:
        """计算地理位置匹配度"""
        # 实现位置匹配逻辑
        pass

class ResultsVisualizer:
    def __init__(self):
        self.console = Console()

    def show_table(self, df: pd.DataFrame, top_n=10):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", justify="center")
        table.add_column("Title", width=30)
        table.add_column("Location", width=20)
        table.add_column("Score", justify="right")
        for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
            table.add_row(
                str(i),
                str(row.get('title', '')),
                str(row.get('location', 'N/A')),
                f"{row['final_score']:.2f}"
            )
        self.console.print(table)

    def plot_eda(self, df: pd.DataFrame):
        plt.figure(figsize=(8, 4))
        sns.histplot(df['final_score'], bins=20, kde=True)
        plt.title('Final Match Score Distribution')
        plt.xlabel('Final Match Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('final_match_score_distribution.png')
        plt.close()
        print("Final match score distribution saved as final_match_score_distribution.png")
        if 'company_profile' in df.columns:
            plt.figure(figsize=(8, 4))
            df['company_profile'].value_counts().head(10).plot(kind='bar')
            plt.title('Top Companies')
            plt.ylabel('Number of Matches')
            plt.tight_layout()
            plt.savefig('top_companies.png')
            plt.close()
            print("Top companies chart saved as top_companies.png")
        if 'location' in df.columns:
            plt.figure(figsize=(8, 4))
            df['location'].value_counts().head(10).plot(kind='bar')
            plt.title('Top Locations')
            plt.ylabel('Number of Matches')
            plt.tight_layout()
            plt.savefig('top_locations.png')
            plt.close()
            print("Top locations chart saved as top_locations.png")

    def show_match_details(self, job_title, match_analysis):
        table = Table(show_header=False)
        table.add_row("Job Title", job_title)
        table.add_row("Skill Match %", f"{int(match_analysis['skill_match_ratio']*100)}%")
        table.add_row("Skill Common", ", ".join(match_analysis["skill_common"]))
        table.add_row("Skill Missing", ", ".join(match_analysis["skill_missing"]))
        table.add_row("Keyword Match %", f"{int(match_analysis['keyword_match_ratio']*100)}%")
        table.add_row("Common Keywords", ", ".join(match_analysis["common_keywords"]))
        table.add_row("Missing Keywords", ", ".join(match_analysis["missing_keywords"]))
        table.add_row("Key Phrases", "\n".join(match_analysis["job_key_phrases"]))
        self.console.print(table)

def average_precision(y_true):
    y_true = np.array(y_true)
    if y_true.sum() == 0:
        return 0.0
    precisions = []
    for i, label in enumerate(y_true, 1):
        if label:
            precisions.append(y_true[:i].sum() / i)
    return np.mean(precisions)

def dcg(y_true, k):
    y_true = np.array(y_true)[:k]
    return np.sum((2**y_true - 1) / np.log2(np.arange(2, len(y_true)+2)))

def ndcg(y_true, k):
    ideal = sorted(y_true, reverse=True)
    return dcg(y_true, k) / (dcg(ideal, k) + 1e-8)

def evaluate_matching(jobs_df, top_ns=[5, 10], label_col='is_true_match'):
    if label_col not in jobs_df.columns:
        print(f"[评估提示] 未检测到真实标签列 `{label_col}`，无法自动评估匹配效果。")
        print(f"如需评估，请在岗位数据中添加一列 `{label_col}`，1表示理想匹配，0表示非理想匹配。")
        return

    true_total = jobs_df[label_col].sum()
    if true_total == 0:
        print(f"[评估提示] 没有标记为理想匹配的岗位，无法评估。")
        return

    y_true = jobs_df[label_col].tolist()
    for n in top_ns:
        topn = jobs_df.head(n)
        hit = int(topn[label_col].sum() > 0)
        precision = topn[label_col].sum() / n
        recall = topn[label_col].sum() / true_total
        ndcg_score = ndcg(y_true, n)
        print(f"Top-{n} Hit Rate: {hit} | Precision@{n}: {precision:.2f} | Recall@{n}: {recall:.2f} | NDCG@{n}: {ndcg_score:.3f}")

    ap = average_precision(y_true)
    print(f"MAP (Mean Average Precision): {ap:.3f}")

def main():
    try:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        resume_path = input("Enter path to your resume (PDF/DOCX): ").strip()
        jobs_path = input("Enter path to job data CSV: ").strip()
        resume_text = ResumeParser.load_resume(resume_path)
        candidate_name = ResumeParser.extract_name(resume_text, resume_path)
        print(f"Candidate Name: {candidate_name}")

        jobs_df = pd.read_csv(jobs_path)
        jobs_df['description'] = jobs_df['description'].fillna('').astype(str)
        matcher = EnhancedJobMatcher()
        # 多模型语义分数
        sim_scores = matcher.compute_similarity(jobs_df, resume_text)
        jobs_df['mpnet_score'] = sim_scores['mpnet']
        jobs_df['minilm_score'] = sim_scores['minilm']
        jobs_df['multi_score'] = sim_scores['multi']

        # 简历技能
        summary = ResumeParser.generate_resume_summary(resume_text)
        resume_skills = summary["skills"]
        resume_skills_set = normalize_skill_set(resume_skills)

        # 岗位技能提取（用关键词近似）
        job_skills_list = []
        for desc in jobs_df['description']:
            job_keywords = matcher.extract_keywords(desc, 15)
            job_skills_list.append(job_keywords)
        jobs_df['job_skills'] = job_skills_list

        # 技能交集分数
        skill_scores = []
        keyword_scores = []
        for idx, row in jobs_df.iterrows():
            job_skills = row['job_skills']
            job_skills_set = normalize_skill_set(job_skills)
            skill_overlap = len(resume_skills_set & job_skills_set) / max(1, len(job_skills_set))
            skill_scores.append(skill_overlap)
            # 关键词交集
            resume_keywords = matcher.extract_keywords(resume_text, 20)
            job_keywords = matcher.extract_keywords(row['description'], 20)
            keyword_overlap = len(set(resume_keywords) & set(job_keywords)) / max(1, len(job_keywords))
            keyword_scores.append(keyword_overlap)
        jobs_df['skill_score'] = skill_scores
        jobs_df['keyword_score'] = keyword_scores

        # 多模型融合+多维度加权
        jobs_df['semantic_score'] = (
            0.5 * jobs_df['mpnet_score'] +
            0.3 * jobs_df['minilm_score'] +
            0.2 * jobs_df['multi_score']
        )
        jobs_df['final_score'] = (
            0.7 * jobs_df['semantic_score'] +
            0.2 * jobs_df['skill_score'] +
            0.1 * jobs_df['keyword_score']
        )
        jobs_df = jobs_df.sort_values('final_score', ascending=False)
        jobs_df.to_csv('top_matches.csv', index=False)
        print("Top matches saved to top_matches.csv")

        # 可视化输出
        visualizer = ResultsVisualizer()
        visualizer.show_table(jobs_df, top_n=10)
        visualizer.plot_eda(jobs_df)

        # 关键信息提取与匹配分析
        print("\n[简历关键信息]")
        print("技能:", summary["skills"])
        print("学历:", summary["education"])
        print("经验:", summary["experience"])

        # 匹配效果评估
        print("\n[匹配效果评估]")
        evaluate_matching(jobs_df, top_ns=[5, 10], label_col='is_true_match')

        # 交互：查看某个岗位的匹配详情
        while True:
            idx = input("\n输入要查看匹配详情的岗位序号(1-10, q退出): ").strip()
            if idx.lower() == 'q':
                break
            if not idx.isdigit() or not (1 <= int(idx) <= 10):
                print("请输入1-10之间的数字或q退出")
                continue
            job = jobs_df.head(10).iloc[int(idx)-1]
            match_analysis = matcher.analyze_match(
                resume_text, job['description'],
                resume_skills=resume_skills,
                job_skills=job['job_skills']
            )
            visualizer.show_match_details(job['title'], match_analysis)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 

